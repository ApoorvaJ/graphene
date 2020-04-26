use crate::*;

pub struct HostVisibleBuffer {
    pub vk_buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
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
            size,
            device: gpu.device.clone(),
        }
    }

    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_buffer(self.vk_buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }

    pub fn upload_data<T>(&self, data: &[T], offset: u64, gpu: &Gpu) {
        let data_size = std::mem::size_of_val(data) as u64;
        debug_assert!(self.size >= offset + data_size);

        unsafe {
            let data_ptr = gpu
                .device
                .map_memory(self.memory, offset, data_size, vk::MemoryMapFlags::empty())
                .expect("Failed to map memory.") as *mut T;

            data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
            gpu.device.unmap_memory(self.memory);
        }
    }
}
