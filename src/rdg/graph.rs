use crate::*;

pub struct BuiltPass {
    pub clear_values: Vec<vk::ClearValue>,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub lambda: Box<dyn FnMut(vk::CommandBuffer)>, // TODO: Investigate async/await
}

pub struct Graph {
    pub device: ash::Device,
    pub render_pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub pipeline_layout: vk::PipelineLayout,
    pub graphics_pipeline: vk::Pipeline,
    pub baked_passes: Vec<BuiltPass>,
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

impl Graph {
    pub fn record_command_buffer(
        &mut self,
        //TODO: Remove these params
        command_buffer: vk::CommandBuffer,
        mesh: &Mesh,
        descriptor_sets: &[vk::DescriptorSet],
        idx: usize,
    ) {
        let baked_pass = &mut self.baked_passes[0];
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording command buffer.");
        }

        let extent = vk::Extent2D {
            width: baked_pass.viewport_width,
            height: baked_pass.viewport_height,
        };

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&baked_pass.clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );

            // Set viewport and scissor
            {
                let viewports = [vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: baked_pass.viewport_width as f32,
                    height: baked_pass.viewport_height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }];
                self.device.cmd_set_viewport(command_buffer, 0, &viewports);

                let scissors = [vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }];
                self.device.cmd_set_scissor(command_buffer, 0, &scissors);
            }
            // Bind index and vertex buffers
            {
                let vertex_buffers = [mesh.vertex_buffer.vk_buffer];
                let offsets = [0_u64];
                self.device
                    .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                self.device.cmd_bind_index_buffer(
                    command_buffer,
                    mesh.index_buffer.vk_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
            }

            // Bind descriptor sets
            {
                let sets = [descriptor_sets[idx]];
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &sets,
                    &[],
                );
            }

            (baked_pass.lambda)(command_buffer);

            self.device.cmd_draw_indexed(
                command_buffer,
                mesh.index_buffer.num_elements as u32,
                1,
                0,
                0,
                0,
            );

            self.device.cmd_end_render_pass(command_buffer);

            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end recording command buffer.");
        }
    }
}
