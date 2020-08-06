use crate::*;

#[derive(Debug, Hash)]
pub struct Pass {
    pub name: String,
    pub vertex_shader: ShaderHandle,
    pub fragment_shader: ShaderHandle,
    pub output_images: Vec<ImageHandle>,
    pub input_image: (vk::ImageView, vk::Sampler), // TODO: Convert to image handle
    pub opt_depth_image: Option<ImageHandle>,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub uniform_buffer: BufferHandle,
}

#[derive(Hash)]
pub struct GraphBuilder {
    pub passes: Vec<(PassHandle, Pass)>,
}

impl GraphBuilder {
    pub fn new() -> GraphBuilder {
        GraphBuilder { passes: Vec::new() }
    }
}

pub struct BuiltPass {
    pub pass_handle: PassHandle,
    pub clear_values: Vec<vk::ClearValue>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,
    pub framebuffer: vk::Framebuffer,
    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    pub graphics_pipeline: vk::Pipeline,
    pub viewport_width: u32,
    pub viewport_height: u32,
}

pub struct Graph {
    device: ash::Device,
    // TODO: What is the correct granularity of this? Should this be shared
    // across the whole context?
    descriptor_pool: vk::DescriptorPool,
    pub built_passes: Vec<BuiltPass>,
    pub shader_handles: Vec<ShaderHandle>, // Needed for shader hot reloading
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            for built_pass in &mut self.built_passes {
                self.device
                    .destroy_pipeline(built_pass.graphics_pipeline, None);
                self.device
                    .destroy_pipeline_layout(built_pass.pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(built_pass.descriptor_set_layout, None);
                self.device
                    .destroy_framebuffer(built_pass.framebuffer, None);
                self.device
                    .destroy_render_pass(built_pass.render_pass, None);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

impl Graph {
    pub fn new(
        graph_builder: GraphBuilder,
        gpu: &Gpu,
        shader_list: &ShaderList,
        buffer_list: &BufferList,
        image_list: &ImageList,
    ) -> Graph {
        // Create descriptor pool
        let descriptor_pool = {
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 2, // TODO: Derive this number
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 2, // TODO: Derive this number
                },
            ];

            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(2) // TODO: Derive this number
                .pool_sizes(&pool_sizes);

            unsafe {
                gpu.device
                    .create_descriptor_pool(&descriptor_pool_create_info, None)
                    .expect("Failed to create descriptor pool.")
            }
        };

        let mut shader_handles = Vec::new();
        let mut built_passes = Vec::new();
        for (pass_handle, pass) in graph_builder.passes {
            /* Record which shader handles have been used. This is needed for
            hot-reloading shaders. */
            shader_handles.push(pass.vertex_shader);
            shader_handles.push(pass.fragment_shader);

            // Find depth image
            let mut opt_depth_image = None;
            if let Some(depth_handle) = pass.opt_depth_image {
                opt_depth_image = Some(
                    image_list
                        .get_image_from_handle(depth_handle)
                        .unwrap_or_else(|| {
                            panic!(
                                "Image with handle `{:?}` not found in the context.",
                                depth_handle
                            )
                        }),
                );
            }

            // Find output images
            let output_images: Vec<&InternalImage> = pass
                .output_images
                .iter()
                .map(|output_handle| {
                    image_list
                        .get_image_from_handle(*output_handle)
                        .unwrap_or_else(|| {
                            panic!(
                                "Image with handle `{:?}` not found in the context.",
                                output_handle
                            )
                        })
                })
                .collect();

            /* Create render pass */
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
                if let Some(depth_image) = opt_depth_image {
                    attachments.push(vk::AttachmentDescription {
                        format: depth_image.image.format,
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
                for output_image in &output_images {
                    attachments.push(vk::AttachmentDescription {
                        format: output_image.image.format,
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
                    gpu.device
                        .create_render_pass(&renderpass_create_info, None)
                        .expect("Failed to create render pass.")
                }
            };

            /* Create framebuffer */
            let framebuffer: vk::Framebuffer = {
                let mut attachments: Vec<vk::ImageView> = Vec::new();
                if let Some(depth_image) = opt_depth_image {
                    attachments.push(depth_image.image.image_view);
                }
                for output_image in &output_images {
                    attachments.push(output_image.image.image_view);
                }

                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(pass.viewport_width)
                    .height(pass.viewport_height)
                    .layers(1);

                unsafe {
                    gpu.device
                        .create_framebuffer(&framebuffer_create_info, None)
                        .expect("Failed to create framebuffer.")
                }
            };

            /* Set clear values */
            let mut clear_values = Vec::new();
            if opt_depth_image.is_some() {
                // Clear value for depth buffer
                clear_values.push(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            }
            for _ in &output_images {
                // Clear values for color buffer
                clear_values.push(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                })
            }

            /* Create descriptor set layout */
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
                    gpu.device
                        .create_descriptor_set_layout(&ubo_layout_create_info, None)
                        .expect("Failed to create Descriptor Set Layout!")
                }
            };

            /* Create descriptor set */
            let descriptor_set = {
                let layouts = [descriptor_set_layout];
                let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts);
                let descriptor_sets = unsafe {
                    gpu.device
                        .allocate_descriptor_sets(&descriptor_set_allocate_info)
                        .expect("Failed to allocate descriptor sets.")
                };

                let uniform_buffer = buffer_list
                    .get_buffer_from_handle(pass.uniform_buffer)
                    .unwrap_or_else(|| {
                        panic!(
                            "Uniform buffer with handle `{:?}` not found in the context.",
                            pass.uniform_buffer
                        )
                    });
                let descriptor_buffer_info = [vk::DescriptorBufferInfo {
                    buffer: uniform_buffer.vk_buffer,
                    offset: 0,
                    range: uniform_buffer.size as u64,
                }];

                let (input_image_view, input_sampler) = pass.input_image;
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
                    gpu.device
                        .update_descriptor_sets(&descriptor_write_sets, &[]);
                }
                descriptor_sets[0]
            };

            /* Create graphics pipeline and pipeline layout */
            let (graphics_pipeline, pipeline_layout) = {
                let main_function_name = CString::new("main").unwrap();
                let vertex_shader = shader_list
                    .get_shader_from_handle(pass.vertex_shader)
                    .unwrap_or_else(|| {
                        panic!(
                            "Vertex shader with handle `{}` not found in the context.",
                            pass.vertex_shader.0
                        )
                    });
                let fragment_shader = shader_list
                    .get_shader_from_handle(pass.fragment_shader)
                    .unwrap_or_else(|| {
                        panic!(
                            "Fragment shader with handle `{}` not found in the context.",
                            pass.fragment_shader.0
                        )
                    });
                let shader_stages = [
                    vk::PipelineShaderStageCreateInfo {
                        stage: vk::ShaderStageFlags::VERTEX,
                        module: vertex_shader.vk_shader_module,
                        p_name: main_function_name.as_ptr(),
                        ..Default::default()
                    },
                    vk::PipelineShaderStageCreateInfo {
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        module: fragment_shader.vk_shader_module,
                        p_name: main_function_name.as_ptr(),
                        ..Default::default()
                    },
                ];

                // (pos: vec3 + normal: vec3) = 6 floats * 4 bytes per float
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

                let set_layouts = [descriptor_set_layout];
                let pipeline_layout_create_info =
                    vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

                let pipeline_layout = unsafe {
                    gpu.device
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
                    gpu.device
                        .create_graphics_pipelines(
                            vk::PipelineCache::null(),
                            &graphic_pipeline_create_infos,
                            None,
                        )
                        .expect("Failed to create Graphics Pipeline.")
                };

                (graphics_pipelines[0], pipeline_layout)
            };

            built_passes.push(BuiltPass {
                pass_handle,
                clear_values,
                descriptor_set_layout,
                descriptor_set,
                framebuffer,
                render_pass,
                pipeline_layout,
                graphics_pipeline,
                viewport_width: pass.viewport_width,
                viewport_height: pass.viewport_height,
            });
        }

        Graph {
            device: gpu.device.clone(),
            descriptor_pool,
            built_passes,
            shader_handles,
        }
    }

    pub fn begin_pass(&self, pass_handle: PassHandle, command_buffer: vk::CommandBuffer) {
        let built_pass = self
            .built_passes
            .iter()
            .find(|&p| p.pass_handle == pass_handle)
            .unwrap_or_else(|| panic!("Pass with handle `{}` not found in graph.", pass_handle.0));

        let extent = vk::Extent2D {
            width: built_pass.viewport_width,
            height: built_pass.viewport_height,
        };

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(built_pass.render_pass)
            .framebuffer(built_pass.framebuffer)
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
                built_pass.graphics_pipeline,
            );

            // Set viewport and scissor
            {
                let viewports = [vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: built_pass.viewport_width as f32,
                    height: built_pass.viewport_height as f32,
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
                    built_pass.pipeline_layout,
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
        }
    }
}
