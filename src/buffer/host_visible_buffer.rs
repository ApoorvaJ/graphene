use crate::*;

pub struct HostVisibleBuffer {
    pub vk_buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    device: ash::Device,
}

impl HostVisibleBuffer {
    pub fn new(size: u64, usage: vk::BufferUsageFlags, gpu: &Gpu) -> HostVisibleBuffer {
        let (vk_buffer, memory) = super::new_raw_buffer(
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            gpu,
        );

        HostVisibleBuffer {
            vk_buffer,
            memory,
            device: gpu.device.clone(),
        }
    }

    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_buffer(self.vk_buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
