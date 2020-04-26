mod platforms;

pub mod apparatus;
pub use apparatus::*;
pub mod buffer;
pub use buffer::*;
pub mod context;
pub use context::*;
pub use context::*;
pub mod facade;
pub use facade::*;
pub mod texture;
pub use texture::*;
pub mod utils;
pub use utils::*;

use ash::version::DeviceV1_0;
use ash::vk;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;

// fn create_buffer(
//     gpu: &Gpu,
//     size: vk::DeviceSize,
//     usage: vk::BufferUsageFlags,
//     required_memory_properties: vk::MemoryPropertyFlags,
// ) -> (vk::Buffer, vk::DeviceMemory) {
// }

pub fn new_buffer<T>(
    data: &[T],
    usage: vk::BufferUsageFlags,
    gpu: &Gpu,
    command_pool: vk::CommandPool,
) -> Buffer {
    let buffer_size = std::mem::size_of_val(data) as u64;

    // ## Create staging buffer in host-visible memory
    let staging_buffer = Buffer::new(
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
                staging_buffer.device_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to map memory.") as *mut T;

        data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        gpu.device.unmap_memory(staging_buffer.device_memory);
    }
    // ## Create buffer in device-local memory
    // TODO: Replace with allocator library?
    let buffer = Buffer::new(
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

            gpu.device.cmd_copy_buffer(
                command_buffer,
                staging_buffer.vk_buffer,
                buffer.vk_buffer,
                &copy_regions,
            );

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

    staging_buffer.destroy();

    buffer
}
