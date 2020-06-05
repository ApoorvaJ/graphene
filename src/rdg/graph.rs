use crate::*;

pub struct BuiltPass {
    pub clear_values: Vec<vk::ClearValue>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,
}

pub struct Graph {
    device: ash::Device,
    descriptor_pool: vk::DescriptorPool,
    pub render_pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub pipeline_layout: vk::PipelineLayout,
    pub graphics_pipeline: vk::Pipeline,
    pub built_passes: Vec<BuiltPass>,
    pub passes: Vec<Pass>,
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            for built_pass in &mut self.built_passes {
                self.device
                    .destroy_descriptor_set_layout(built_pass.descriptor_set_layout, None);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

impl Graph {
    pub fn new(graph_builder: GraphBuilder) -> Graph {
        let device = graph_builder.device;
        let pass = &graph_builder.passes[0];

        // # Create render pass
        let render_pass = {
            let mut attachments: Vec<vk::AttachmentDescription> = Vec::new();
            let mut attachment_idx = 0;
            let mut depth_attachment_ptr = ptr::null();
            let mut color_attachments = Vec::new();

            // Depth attachment description and reference
            let depth_attachment = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };
            if let Some((_depth_image_view, depth_format)) = pass.opt_depth {
                attachments.push(vk::AttachmentDescription {
                    format: depth_format,
                    flags: vk::AttachmentDescriptionFlags::empty(),
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::DONT_CARE, // TODO: Derive from graph
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                });

                depth_attachment_ptr = &depth_attachment;
                attachment_idx += 1;
            }

            // Color attachment descriptions and references
            for (_image_view, format) in &pass.outputs {
                attachments.push(vk::AttachmentDescription {
                    format: *format,
                    flags: vk::AttachmentDescriptionFlags::empty(),
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE, // TODO: Derive from graph
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                });
                color_attachments.push(vk::AttachmentReference {
                    attachment: attachment_idx,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                });
                attachment_idx += 1;
            }

            let subpasses = [vk::SubpassDescription {
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                color_attachment_count: 1,
                p_color_attachments: color_attachments.as_ptr(),
                p_depth_stencil_attachment: depth_attachment_ptr,
                ..Default::default()
            }];

            let renderpass_create_info = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses);

            unsafe {
                device
                    .create_render_pass(&renderpass_create_info, None)
                    .expect("Failed to create render pass!")
            }
        };

        // # Create framebuffer
        let framebuffer: vk::Framebuffer = {
            let mut attachments: Vec<vk::ImageView> = Vec::new();
            if let Some((depth_image_view, _depth_format)) = pass.opt_depth {
                attachments.push(depth_image_view);
            }
            for (image_view, _format) in &pass.outputs {
                attachments.push(*image_view);
            }

            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(pass.viewport_width)
                .height(pass.viewport_height)
                .layers(1);

            unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .expect("Failed to create Framebuffer!")
            }
        };

        // # Create descriptor pool
        let descriptor_pool = {
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1, // TODO: Derive this number
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1, // TODO: Derive this number
                },
            ];

            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1) // TODO: Derive this number
                .pool_sizes(&pool_sizes);

            unsafe {
                device
                    .create_descriptor_pool(&descriptor_pool_create_info, None)
                    .expect("Failed to create descriptor pool.")
            }
        };

        // Bake passes
        let built_passes = {
            let mut clear_values = Vec::new();
            // Clear value for depth buffer
            if pass.opt_depth.is_some() {
                clear_values.push(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            }
            // Clear values for color buffer
            for _ in &pass.outputs {
                clear_values.push(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                })
            }

            // Descriptor set layout
            let descriptor_set_layout = {
                let bindings = [
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        p_immutable_samplers: ptr::null(),
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        p_immutable_samplers: ptr::null(),
                    },
                ];

                let ubo_layout_create_info =
                    vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

                unsafe {
                    device
                        .create_descriptor_set_layout(&ubo_layout_create_info, None)
                        .expect("Failed to create Descriptor Set Layout!")
                }
            };

            // Descriptor set
            let descriptor_set = {
                let layouts = [descriptor_set_layout];
                let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts);
                let descriptor_sets = unsafe {
                    device
                        .allocate_descriptor_sets(&descriptor_set_allocate_info)
                        .expect("Failed to allocate descriptor sets.")
                };
                let (vk_buffer, buffer_size) = pass.buffer_info;
                let descriptor_buffer_info = [vk::DescriptorBufferInfo {
                    buffer: vk_buffer,
                    offset: 0,
                    range: buffer_size as u64,
                }];

                let (input_image_view, input_sampler) = pass.input_texture;
                let descriptor_image_info = [vk::DescriptorImageInfo {
                    sampler: input_sampler,
                    image_view: input_image_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];

                let descriptor_write_sets = [
                    vk::WriteDescriptorSet {
                        dst_set: descriptor_sets[0],
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: descriptor_buffer_info.as_ptr(),
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: descriptor_sets[0],
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: descriptor_image_info.as_ptr(),
                        ..Default::default()
                    },
                ];

                unsafe {
                    device.update_descriptor_sets(&descriptor_write_sets, &[]);
                }
                descriptor_sets[0]
            };

            let built_passes = vec![BuiltPass {
                clear_values,
                descriptor_set_layout,
                descriptor_set,
            }];
            built_passes
        };

        // # Create graphics pipeline
        let (graphics_pipeline, pipeline_layout) = {
            let main_function_name = CString::new("main").unwrap();
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: pass.shader_modules[0],
                    p_name: main_function_name.as_ptr(),
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: pass.shader_modules[1],
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

            // Initialize to defaults. It will be ignored because pipeline viewport is dynamic.
            let viewports = [vk::Viewport {
                ..Default::default()
            }];

            // Initialize to defaults. It will be ignored because pipeline scissor is dynamic.
            let scissors = [vk::Rect2D {
                ..Default::default()
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

            let set_layouts = [built_passes[0].descriptor_set_layout];
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

            let pipeline_layout = unsafe {
                device
                    .create_pipeline_layout(&pipeline_layout_create_info, None)
                    .expect("Failed to create pipeline layout.")
            };

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                p_next: ptr::null(),
                flags: ash::vk::PipelineDynamicStateCreateFlags::empty(),
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
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
                p_dynamic_state: &dynamic_state_create_info, // No dynamic state
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                ..Default::default()
            }];

            let graphics_pipelines = unsafe {
                device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &graphic_pipeline_create_infos,
                        None,
                    )
                    .expect("Failed to create Graphics Pipeline.")
            };

            (graphics_pipelines[0], pipeline_layout)
        };

        Graph {
            device,
            descriptor_pool,
            render_pass,
            framebuffer,
            graphics_pipeline,
            pipeline_layout,
            built_passes,
            passes: graph_builder.passes,
        }
    }

    pub fn begin_pass(&self, _pass_handle: u64, command_buffer: vk::CommandBuffer) {
        let pass = &self.passes[0];
        let built_pass = &self.built_passes[0];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording command buffer.");
        }

        let extent = vk::Extent2D {
            width: pass.viewport_width,
            height: pass.viewport_height,
        };

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&built_pass.clear_values);

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
                    width: pass.viewport_width as f32,
                    height: pass.viewport_height as f32,
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
            // Bind descriptor sets
            {
                let sets = [built_pass.descriptor_set];
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &sets,
                    &[],
                );
            }
        }
    }

    pub fn end_pass(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_end_render_pass(command_buffer);

            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end recording command buffer.");
        }
    }
}
