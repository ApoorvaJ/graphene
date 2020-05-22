use crate::*;

pub struct GraphBuilder<'a> {
    device: ash::Device,
    passes: Vec<Pass<'a>>,
}

impl<'a> GraphBuilder<'a> {
    pub fn new(gpu: &Gpu) -> GraphBuilder {
        GraphBuilder {
            device: gpu.device.clone(),
            passes: Vec::new(),
        }
    }

    pub fn add_pass(mut self, pass: Pass<'a>) -> GraphBuilder<'a> {
        self.passes.push(pass);
        self
    }

    pub fn build(
        self,
        // TODO: Remove these params
        shader_modules: &Vec<vk::ShaderModule>,
        uniform_buffer_layout: vk::DescriptorSetLayout,
    ) -> Graph {
        let pass = &self.passes[0];

        // # Create render pass
        let render_pass = {
            let mut attachments: Vec<vk::AttachmentDescription> = Vec::new();
            if let Some(depth_tex) = pass.opt_output_depth {
                attachments.push(vk::AttachmentDescription {
                    format: depth_tex.format,
                    flags: vk::AttachmentDescriptionFlags::empty(),
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::DONT_CARE, // TODO: Derive from graph
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                });
            }
            for color_tex in &pass.outputs_color {
                attachments.push(vk::AttachmentDescription {
                    format: color_tex.format,
                    flags: vk::AttachmentDescriptionFlags::empty(),
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE, // TODO: Derive from graph
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                });
            }

            let depth_attachment_ref = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };
            let color_attachment_ref = [vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];

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
                self.device
                    .create_render_pass(&renderpass_create_info, None)
                    .expect("Failed to create render pass!")
            }
        };

        // # Create framebuffer
        let framebuffer: vk::Framebuffer = {
            let (w, h) = if let Some(depth_tex) = pass.opt_output_depth {
                (depth_tex.width, depth_tex.height)
            } else {
                assert!(
                    pass.outputs_color.len() > 0,
                    "At least a depth texture or a single color texture needs to be bound."
                );
                (pass.outputs_color[0].width, pass.outputs_color[0].height)
            };
            // TODO: Assert that color and depth textures have the same resolution
            let mut attachments: Vec<vk::ImageView> = Vec::new();
            if let Some(depth_tex) = pass.opt_output_depth {
                attachments.push(depth_tex.image_view);
            }
            for color_tex in &pass.outputs_color {
                attachments.push(color_tex.image_view);
            }

            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(w)
                .height(h)
                .layers(1);

            unsafe {
                self.device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .expect("Failed to create Framebuffer!")
            }
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

            let set_layouts = [uniform_buffer_layout];
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

            let pipeline_layout = unsafe {
                self.device
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
                self.device
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
            device: self.device,
            render_pass,
            framebuffer,
            graphics_pipeline,
            pipeline_layout,
        }
    }
}
