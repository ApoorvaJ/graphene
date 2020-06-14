use crate::*;

pub mod device_local_buffer;
pub use device_local_buffer::*;
pub mod host_visible_buffer;
pub use host_visible_buffer::*;

fn new_raw_buffer(
    size: usize,
    usage: vk::BufferUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
    gpu: &Gpu,
) -> (vk::Buffer, vk::DeviceMemory) {
    // Create buffer
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size as vk::DeviceSize)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let vk_buffer = unsafe {
        gpu.device
            .create_buffer(&buffer_create_info, None)
            .expect("Failed to create buffer.")
    };
    // Locate memory type
    let mem_requirements = unsafe { gpu.device.get_buffer_memory_requirements(vk_buffer) };
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

    let device_memory = unsafe {
        gpu.device
            .allocate_memory(&allocate_info, None)
            .expect("Failed to allocate buffer memory.")
    };
    // Bind memory to buffer
    unsafe {
        gpu.device
            .bind_buffer_memory(vk_buffer, device_memory, 0)
            .expect("Failed to bind buffer.");
    }

    (vk_buffer, device_memory)
}
