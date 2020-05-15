use crate::*;

pub struct RenderGraph {
    _device: ash::Device,
    pub command_buffer: vk::CommandBuffer,
}

impl RenderGraph {
    pub fn new(command_pool: vk::CommandPool, gpu: &Gpu) -> RenderGraph {
        let device = gpu.device.clone();

        // # Allocate command buffers
        let command_buffers = {
            let info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            unsafe {
                device
                    .allocate_command_buffers(&info)
                    .expect("Failed to allocate command buffer.")
            }
        };

        RenderGraph {
            _device: device,
            command_buffer: command_buffers[0],
        }
    }
}
