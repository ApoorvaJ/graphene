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
                    size: size,
                }];

                gpu.device.cmd_copy_buffer(
                    command_buffer,
                    staging_buffer.vk_buffer,
                    vk_buffer,
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
