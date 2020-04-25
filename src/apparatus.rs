use crate::*;

pub struct Apparatus {
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    // - Pipelines
    pub pipeline_layout: vk::PipelineLayout,
    pub graphics_pipeline: vk::Pipeline,
    // - Commands
    pub command_buffers: Vec<vk::CommandBuffer>,
}

impl Apparatus {
    pub fn new(
        gpu: &Gpu,
        facade: &Facade,
        command_pool: vk::CommandPool,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        num_indices: u32,
        uniform_buffer_layout: vk::DescriptorSetLayout,
        descriptor_sets: &[vk::DescriptorSet],
        shader_modules: Vec<vk::ShaderModule>,
    ) -> Apparatus {
        // # Create render pass
        let render_pass = {
            let attachments = vec![
                // Color attachment
                vk::AttachmentDescription {
                    format: facade.swapchain_format,
                    flags: vk::AttachmentDescriptionFlags::empty(),
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                },
                // Depth attachment
                vk::AttachmentDescription {
                    format: facade.depth_texture.format,
                    flags: vk::AttachmentDescriptionFlags::empty(),
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::DONT_CARE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE, // ?
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE, // ?
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                },
            ];

            let color_attachment_ref = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let depth_attachment_ref = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };

            let subpasses = [vk::SubpassDescription {
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                color_attachment_count: 1,
                p_color_attachments: color_attachment_ref.as_ptr(),
                p_depth_stencil_attachment: &depth_attachment_ref,
                ..Default::default()
            }];

            let renderpass_create_info = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses);

            unsafe {
                gpu.device
                    .create_render_pass(&renderpass_create_info, None)
                    .expect("Failed to create render pass!")
            }
        };

        // # Create framebuffers
        let framebuffers: Vec<vk::Framebuffer> = {
            facade
                .swapchain_imageviews
                .iter()
                .map(|&imageview| {
                    let attachments = [imageview, facade.depth_image_view];

                    let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(render_pass)
                        .attachments(&attachments)
                        .width(facade.swapchain_extent.width)
                        .height(facade.swapchain_extent.height)
                        .layers(1);

                    unsafe {
                        gpu.device
                            .create_framebuffer(&framebuffer_create_info, None)
                            .expect("Failed to create Framebuffer!")
                    }
                })
                .collect()
        };

        // # Create graphics pipeline
        let (graphics_pipeline, pipeline_layout) = {
            let main_function_name = CString::new("main").unwrap();
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: shader_modules[0],
                    p_name: main_function_name.as_ptr(),
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: shader_modules[1],
                    p_name: main_function_name.as_ptr(),
                    ..Default::default()
                },
            ];

            // (pos: vec3 + color: vec3) = 6 floats * 4 bytes per float
            const VERTEX_STRIDE: u32 = 24;
            let binding_descriptions = [vk::VertexInputBindingDescription {
                binding: 0,
                stride: VERTEX_STRIDE,
                ..Default::default()
            }];
            let attribute_descriptions = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 12,
                },
            ];
            let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: binding_descriptions.len() as u32,
                p_vertex_binding_descriptions: binding_descriptions.as_ptr(),
                vertex_attribute_description_count: attribute_descriptions.len() as u32,
                p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
                ..Default::default()
            };

            let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };

            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: facade.swapchain_extent.width as f32,
                height: facade.swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];

            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: facade.swapchain_extent,
            }];

            let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
                scissor_count: scissors.len() as u32,
                p_scissors: scissors.as_ptr(),
                viewport_count: viewports.len() as u32,
                p_viewports: viewports.as_ptr(),
                ..Default::default()
            };

            let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::BACK,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            };

            let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::TRUE,
                depth_write_enable: vk::TRUE,
                depth_compare_op: vk::CompareOp::LESS,
                max_depth_bounds: 1.0,
                min_depth_bounds: 0.0,
                ..Default::default()
            };

            let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                color_write_mask: vk::ColorComponentFlags::all(),
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ZERO,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
            }];

            let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
                attachment_count: color_blend_attachment_states.len() as u32,
                p_attachments: color_blend_attachment_states.as_ptr(),
                blend_constants: [0.0, 0.0, 0.0, 0.0],
                ..Default::default()
            };

            let set_layouts = [uniform_buffer_layout];
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

            let pipeline_layout = unsafe {
                gpu.device
                    .create_pipeline_layout(&pipeline_layout_create_info, None)
                    .expect("Failed to create pipeline layout.")
            };

            let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
                stage_count: shader_stages.len() as u32,
                p_stages: shader_stages.as_ptr(),
                p_vertex_input_state: &vertex_input_state_create_info,
                p_input_assembly_state: &vertex_input_assembly_state_info,
                p_tessellation_state: ptr::null(),
                p_viewport_state: &viewport_state_create_info,
                p_rasterization_state: &rasterization_state_create_info,
                p_multisample_state: &multisample_state_create_info,
                p_depth_stencil_state: &depth_state_create_info,
                p_color_blend_state: &color_blend_state,
                p_dynamic_state: ptr::null(), // No dynamic state
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                ..Default::default()
            }];

            let graphics_pipelines = unsafe {
                gpu.device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &graphic_pipeline_create_infos,
                        None,
                    )
                    .expect("Failed to create Graphics Pipeline.")
            };

            unsafe {
                for stage in shader_modules {
                    gpu.device.destroy_shader_module(stage, None);
                }
            }

            (graphics_pipelines[0], pipeline_layout)
        };

        // # Allocate command buffers
        let command_buffers = {
            let info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(facade.num_frames as u32);

            unsafe {
                gpu.device
                    .allocate_command_buffers(&info)
                    .expect("Failed to allocate command buffer.")
            }
        };

        // # Record command buffers
        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

            unsafe {
                gpu.device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording command buffer.");
            }

            let clear_values = [
                vk::ClearValue {
                    // Clear value for color buffer
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    // Clear value for depth buffer
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass)
                .framebuffer(framebuffers[i])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: facade.swapchain_extent,
                })
                .clear_values(&clear_values);

            unsafe {
                gpu.device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                gpu.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                );

                // Bind index and vertex buffers
                {
                    let vertex_buffers = [vertex_buffer];
                    let offsets = [0_u64];
                    gpu.device.cmd_bind_vertex_buffers(
                        command_buffer,
                        0,
                        &vertex_buffers,
                        &offsets,
                    );
                    gpu.device.cmd_bind_index_buffer(
                        command_buffer,
                        index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

                // Bind descriptor sets
                {
                    let sets = [descriptor_sets[i]];
                    gpu.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &sets,
                        &[],
                    );
                }

                gpu.device
                    .cmd_draw_indexed(command_buffer, num_indices, 1, 0, 0, 0);

                gpu.device.cmd_end_render_pass(command_buffer);

                gpu.device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to end recording command buffer.");
            }
        }

        Apparatus {
            render_pass,
            framebuffers,
            graphics_pipeline,
            pipeline_layout,
            command_buffers,
        }
    }

    pub fn destroy(&self, gpu: &Gpu) {
        unsafe {
            gpu.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            gpu.device.destroy_pipeline(self.graphics_pipeline, None);
            for &framebuffer in self.framebuffers.iter() {
                gpu.device.destroy_framebuffer(framebuffer, None);
            }
            gpu.device.destroy_render_pass(self.render_pass, None);
        }
    }
}
