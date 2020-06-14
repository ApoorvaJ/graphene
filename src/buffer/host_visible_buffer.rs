use crate::*;

pub struct HostVisibleBuffer {
    pub vk_buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: usize,
    device: ash::Device,
}

impl Drop for HostVisibleBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.vk_buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl HostVisibleBuffer {
    pub fn new(size: usize, usage: vk::BufferUsageFlags, gpu: &Gpu) -> HostVisibleBuffer {
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

    pub fn upload_data<T>(&self, data: &[T], offset: usize, gpu: &Gpu) {
        let data_size = std::mem::size_of_val(data);
        debug_assert!(self.size >= offset + data_size);

        unsafe {
            let data_ptr = gpu
                .device
                .map_memory(
                    self.memory,
                    offset as u64,
                    data_size as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map memory.") as *mut T;

            data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
            gpu.device.unmap_memory(self.memory);
        }
    }
}
