use crate::*;

pub struct Buffer {
    pub vk_buffer: vk::Buffer,
    pub device_memory: vk::DeviceMemory,
    device: ash::Device,
}

impl Buffer {
    pub fn new(
        gpu: &Gpu,
        size: u64,
        usage: vk::BufferUsageFlags,
        required_memory_properties: vk::MemoryPropertyFlags,
    ) -> Buffer {
        let device = gpu.device.clone();

        // Create buffer
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size as vk::DeviceSize)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vk_buffer = unsafe {
            device
                .create_buffer(&buffer_create_info, None)
                .expect("Failed to create buffer.")
        };
        // Locate memory type
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(vk_buffer) };
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
            device
                .allocate_memory(&allocate_info, None)
                .expect("Failed to allocate buffer memory.")
        };
        // Bind memory to buffer
        unsafe {
            device
                .bind_buffer_memory(vk_buffer, device_memory, 0)
                .expect("Failed to bind buffer.");
        }

        Buffer {
            vk_buffer,
            device_memory,
            device,
        }
    }

    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_buffer(self.vk_buffer, None);
            self.device.free_memory(self.device_memory, None);
        }
    }
}
