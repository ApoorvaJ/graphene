use crate::*;

pub struct DeviceLocalBuffer {
    pub vk_buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    device: ash::Device,
}

impl DeviceLocalBuffer {
    pub fn new<T>(
        data: &[T],
        usage: vk::BufferUsageFlags,
        gpu: &Gpu,
        command_pool: vk::CommandPool,
    ) -> DeviceLocalBuffer {
        let size = std::mem::size_of_val(data) as u64;

        // ## Create staging buffer in host-visible memory
        let staging_buffer = HostVisibleBuffer::new(size, vk::BufferUsageFlags::TRANSFER_SRC, &gpu);

        // ## Copy data to staging buffer
        staging_buffer.upload_data(data, 0, gpu);

        // ## Create buffer in device-local memory
        let (vk_buffer, memory) = super::new_raw_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            gpu,
        );

        // ## Copy staging buffer -> vertex buffer
        {
            let command_buffer = begin_single_use_command_buffer(&gpu.device, command_pool);

            unsafe {
                let copy_regions = [vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: size,
                }];

                gpu.device.cmd_copy_buffer(
                    command_buffer,
                    staging_buffer.vk_buffer,
                    vk_buffer,
                    &copy_regions,
                );
            }

            end_single_use_command_buffer(command_buffer, command_pool, &gpu);
        }

        DeviceLocalBuffer {
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
