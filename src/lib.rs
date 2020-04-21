pub mod apparatus;
pub use apparatus::*;
pub mod context;
pub use context::*;
pub use context::*;
pub mod facade;
pub use facade::*;
mod platforms;

use ash::version::DeviceV1_0;
use ash::vk;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;

const NUM_FRAMES: usize = 2;

fn create_buffer(
    gpu: &Gpu,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let device = &gpu.device;
    // Create buffer
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&buffer_create_info, None)
            .expect("Failed to create buffer.")
    };
    // Locate memory type
    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = gpu
        .memory_properties
        .memory_types
        .iter()
        .enumerate()
        .position(|(i, &m)| {
            (mem_requirements.memory_type_bits & (1 << i)) > 0
                && m.property_flags.contains(required_memory_properties)
        })
        .expect("Failed to find suitable memory type.") as u32;
    // Allocate memory
    // TODO: Replace with allocator library?
    let allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let buffer_memory = unsafe {
        device
            .allocate_memory(&allocate_info, None)
            .expect("Failed to allocate buffer memory.")
    };
    // Bind memory to buffer
    unsafe {
        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .expect("Failed to bind buffer.");
    }

    (buffer, buffer_memory)
}

pub fn new_buffer<T>(
    data: &[T],
    usage: vk::BufferUsageFlags,
    gpu: &Gpu,
    command_pool: vk::CommandPool,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;

    // ## Create staging buffer in host-visible memory
    // TODO: Replace with allocator library?
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        &gpu,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );
    // ## Copy data to staging buffer
    unsafe {
        let data_ptr = gpu
            .device
            .map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to map memory.") as *mut T;

        data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        gpu.device.unmap_memory(staging_buffer_memory);
    }
    // ## Create buffer in device-local memory
    // TODO: Replace with allocator library?
    let (buffer, buffer_memory) = create_buffer(
        &gpu,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    // ## Copy staging buffer -> vertex buffer
    {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe {
            gpu.device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate command buffer.")
        };
        let command_buffer = command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            gpu.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer.");

            let copy_regions = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: buffer_size,
            }];

            gpu.device
                .cmd_copy_buffer(command_buffer, staging_buffer, buffer, &copy_regions);

            gpu.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer");
        }

        let submit_info = [vk::SubmitInfo {
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            ..Default::default()
        }];

        unsafe {
            gpu.device
                .queue_submit(gpu.graphics_queue, &submit_info, vk::Fence::null())
                .expect("Failed to Submit Queue.");
            gpu.device
                .queue_wait_idle(gpu.graphics_queue)
                .expect("Failed to wait Queue idle");

            gpu.device
                .free_command_buffers(command_pool, &command_buffers);
        }
    }

    unsafe {
        gpu.device.destroy_buffer(staging_buffer, None);
        gpu.device.free_memory(staging_buffer_memory, None);
    }

    (buffer, buffer_memory)
}
