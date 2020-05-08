use crate::*;

use glam::*;
use std::os::raw::c_void;
use std::ptr;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

pub struct Context {
    window: winit::window::Window,
    event_loop: Option<winit::event_loop::EventLoop<()>>,

    pub gpu: Gpu,
    pub facade: Facade, // Resolution-dependent apparatus
    pub apparatus: Apparatus,
    pub command_pool: vk::CommandPool,
    pub vertex_buffer: DeviceLocalBuffer,
    pub index_buffer: DeviceLocalBuffer,
    pub num_indices: u32,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub uniform_buffer_layout: vk::DescriptorSetLayout,
    pub uniform_buffers: Vec<HostVisibleBuffer>,
    pub environment_texture: Texture,
    pub environment_sampler: vk::Sampler,

    pub current_frame: usize,
    start_instant: std::time::Instant,

    _watcher: notify::RecommendedWatcher, // Need to keep this alive to keep the receiver alive
    watch_rx: std::sync::mpsc::Receiver<notify::DebouncedEvent>,

    basis: Basis,
}

impl Context {
    pub fn recreate_resolution_dependent_state(&mut self) {
        unsafe {
            self.gpu
                .device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
        self.apparatus.destroy(&self.gpu);
        self.facade.destroy(&self.gpu);
        self.facade = Facade::new(&self.basis, &self.gpu, &self.window);
        let (shader_modules, _) =
            utils::get_shader_modules(&self.gpu).expect("Failed to load shader modules");
        self.apparatus = Apparatus::new(
            &self.gpu,
            &self.facade,
            self.command_pool,
            &self.vertex_buffer,
            &self.index_buffer,
            self.num_indices,
            self.uniform_buffer_layout,
            &self.descriptor_sets,
            shader_modules,
        );
    }

    pub fn new(uniform_buffer_size: usize) -> Context {
        const APP_NAME: &str = "";

        // # Init window
        let event_loop = EventLoop::new();
        let window = {
            winit::window::WindowBuilder::new()
                .with_title(APP_NAME)
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
                .build(&event_loop)
                .expect("Failed to create window.")
        };

        let basis = Basis::new(APP_NAME, &window);
        let gpu = Gpu::new(&basis);

        // # Create command pool
        let command_pool = {
            let info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(gpu.graphics_queue_idx);

            unsafe {
                gpu.device
                    .create_command_pool(&info, None)
                    .expect("Failed to create command pool")
            }
        };

        // # Load mesh
        // TODO: Benchmark and optimize
        let (vertices_data, indices_data) = {
            let mut vertices_data: Vec<f32> = Vec::new();
            let mut indices_data: Vec<u32> = Vec::new();

            let (gltf, buffers, _) =
                gltf::import("assets/meshes/suzanne.glb").expect("Failed to open mesh.");
            for mesh in gltf.meshes() {
                for primitive in mesh.primitives() {
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                    if let Some(iter_pos) = reader.read_positions() {
                        if let Some(iter_norm) = reader.read_normals() {
                            for (pos, norm) in iter_pos.zip(iter_norm) {
                                vertices_data.extend_from_slice(&pos);
                                vertices_data.extend_from_slice(&norm);
                            }
                        }
                    }
                    if let Some(iter) = reader.read_indices() {
                        match iter {
                            gltf::mesh::util::ReadIndices::U8(iter_2) => {
                                for idx in iter_2 {
                                    indices_data.push(idx as u32);
                                }
                            }
                            gltf::mesh::util::ReadIndices::U16(iter_2) => {
                                for idx in iter_2 {
                                    indices_data.push(idx as u32);
                                }
                            }
                            gltf::mesh::util::ReadIndices::U32(iter_2) => {
                                for idx in iter_2 {
                                    indices_data.push(idx as u32);
                                }
                            }
                        }
                    }
                }
            }

            (vertices_data, indices_data)
        };

        // # Create and upload the vertex buffer
        let vertex_buffer = DeviceLocalBuffer::new(
            &vertices_data,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &gpu,
            command_pool,
        );

        // # Create and upload index buffer
        let index_buffer = DeviceLocalBuffer::new(
            &indices_data,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &gpu,
            command_pool,
        );

        // TODO: Move this up?
        let facade = Facade::new(&basis, &gpu, &window);

        // # Uniform buffer descriptor layout
        let uniform_buffer_layout = {
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

        // # Create the uniform buffer
        let uniform_buffers: Vec<HostVisibleBuffer> = (0..facade.num_frames)
            .map(|_| {
                HostVisibleBuffer::new(
                    uniform_buffer_size as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    &gpu,
                )
            })
            .collect();

        // # Environment texture
        let environment_texture = Texture::new_from_image(
            std::path::Path::new("assets/textures/env_carpentry_shop_02_2k.jpg"),
            &gpu,
            command_pool,
        );

        // TODO: Spin this out into separate struct/function
        let environment_sampler = {
            let sampler_create_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true) // TODO: Disable this by default?
                .max_anisotropy(16.0) //
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK);

            unsafe {
                gpu.device
                    .create_sampler(&sampler_create_info, None)
                    .expect("Failed to create Sampler!")
            }
        };

        // # Create descriptor pool
        let descriptor_pool = {
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: facade.num_frames as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: facade.num_frames as u32,
                },
            ];

            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(facade.num_frames as u32)
                .pool_sizes(&pool_sizes);

            unsafe {
                gpu.device
                    .create_descriptor_pool(&descriptor_pool_create_info, None)
                    .expect("Failed to create descriptor pool.")
            }
        };

        // # Create descriptor sets
        let descriptor_sets = {
            // TODO: Convert to collect pattern
            let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
            for _ in 0..facade.num_frames {
                layouts.push(uniform_buffer_layout);
            }

            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);

            let descriptor_sets = unsafe {
                gpu.device
                    .allocate_descriptor_sets(&descriptor_set_allocate_info)
                    .expect("Failed to allocate descriptor sets.")
            };

            for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
                let descriptor_buffer_info = [vk::DescriptorBufferInfo {
                    buffer: uniform_buffers[i].vk_buffer,
                    offset: 0,
                    range: uniform_buffer_size as u64,
                }];

                let descriptor_image_info = [vk::DescriptorImageInfo {
                    sampler: environment_sampler,
                    image_view: environment_texture.image_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];

                let descriptor_write_sets = [
                    vk::WriteDescriptorSet {
                        dst_set: descriptor_set,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: descriptor_buffer_info.as_ptr(),
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: descriptor_set,
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
            }

            descriptor_sets
        };

        let (shader_modules, _) =
            utils::get_shader_modules(&gpu).expect("Failed to load shader modules");
        let apparatus = Apparatus::new(
            &gpu,
            &facade,
            command_pool,
            &vertex_buffer,
            &index_buffer,
            indices_data.len() as u32,
            uniform_buffer_layout,
            &descriptor_sets,
            shader_modules,
        );

        // Add expect messages to all these unwraps
        let (watcher, watch_rx) = {
            use notify::{RecommendedWatcher, RecursiveMode, Watcher};
            use std::sync::mpsc::channel;
            use std::time::Duration;

            let (tx, rx) = channel();
            let mut watcher: RecommendedWatcher = Watcher::new(tx, Duration::from_secs(2)).unwrap();
            watcher.watch("./assets", RecursiveMode::Recursive).unwrap();
            (watcher, rx)
        };

        Context {
            window,
            event_loop: Some(event_loop),

            // - Device
            gpu,
            facade,
            apparatus,
            command_pool,
            vertex_buffer,
            index_buffer,
            num_indices: indices_data.len() as u32,
            descriptor_pool,
            descriptor_sets,
            uniform_buffer_layout,
            uniform_buffers,
            environment_texture,
            environment_sampler,

            current_frame: 0,
            start_instant: std::time::Instant::now(),

            _watcher: watcher,
            watch_rx,
            basis,
        }
    }

    fn draw_frame<F>(&mut self, on_draw: &mut F)
    where
        F: FnMut(&mut Context, f32, usize),
    {
        let wait_fences = [self.facade.command_buffer_complete_fences[self.current_frame]];

        let (image_index, _is_sub_optimal) = unsafe {
            self.gpu
                .device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence.");

            let result = self.facade.ext_swapchain.acquire_next_image(
                self.facade.swapchain,
                std::u64::MAX,
                self.facade.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );
            match result {
                Ok(image_idx) => image_idx,
                Err(error_code) => {
                    match error_code {
                        vk::Result::ERROR_OUT_OF_DATE_KHR => {
                            // Window is resized. Recreate the swapchain
                            // and exit early without drawing this frame.
                            self.recreate_resolution_dependent_state();
                            return;
                        }
                        _ => panic!("Failed to acquire swapchain image."),
                    }
                }
            }
        };

        let elapsed_seconds = self.start_instant.elapsed().as_secs_f32();
        on_draw(self, elapsed_seconds, image_index as usize);

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_semaphores = [self.facade.image_available_semaphores[self.current_frame]];
        let signal_semaphores = [self.facade.render_finished_semaphores[self.current_frame]];
        let command_buffers = [self.apparatus.command_buffers[image_index as usize]];

        let submit_infos = [vk::SubmitInfo {
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        }];

        unsafe {
            self.gpu
                .device
                .reset_fences(&wait_fences)
                .expect("Failed to reset fence.");

            self.gpu
                .device
                .queue_submit(
                    self.gpu.graphics_queue,
                    &submit_infos,
                    self.facade.command_buffer_complete_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.facade.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        // Present the queue
        {
            let result = unsafe {
                self.facade
                    .ext_swapchain
                    .queue_present(self.gpu.present_queue, &present_info)
            };

            if let Err(error_code) = result {
                match error_code {
                    vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                        // Window is resized. Recreate the swapchain
                        self.recreate_resolution_dependent_state();
                    }
                    _ => panic!("Failed to present queue."),
                }
            }
        }

        for event in self.watch_rx.try_iter() {
            use notify::DebouncedEvent::*;
            match event {
                Write(_) | Remove(_) | Rename(_, _) => {
                    unsafe {
                        self.gpu
                            .device
                            .device_wait_idle()
                            .expect("Failed to wait device idle!");
                    }
                    if let Some((shader_modules, num_changed)) =
                        utils::get_shader_modules(&self.gpu)
                    {
                        if num_changed > 0 {
                            self.apparatus.destroy(&self.gpu);
                            self.apparatus = Apparatus::new(
                                &self.gpu,
                                &self.facade,
                                self.command_pool,
                                &self.vertex_buffer,
                                &self.index_buffer,
                                self.num_indices,
                                self.uniform_buffer_layout,
                                &self.descriptor_sets,
                                shader_modules,
                            );
                        }
                    }
                }
                _ => (),
            }
        }

        self.current_frame = (self.current_frame + 1) % self.facade.num_frames;
    }

    pub fn run_loop<F: 'static>(mut self, mut on_draw: F)
    where
        F: FnMut(&mut Context, f32, usize),
    {
        self.event_loop
            .take()
            .unwrap()
            .run(move |event, _, control_flow| match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        if self.facade.swapchain_extent.width != physical_size.width
                            || self.facade.swapchain_extent.height != physical_size.height
                        {
                            self.recreate_resolution_dependent_state();
                        }
                    }
                    WindowEvent::KeyboardInput { input, .. } => match input {
                        KeyboardInput {
                            virtual_keycode,
                            state,
                            ..
                        } => match (virtual_keycode, state) {
                            (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                *control_flow = ControlFlow::Exit
                            }
                            (Some(VirtualKeyCode::Return), ElementState::Pressed) => {
                                *control_flow = ControlFlow::Exit
                            }
                            _ => {}
                        },
                    },
                    _ => {}
                },
                Event::MainEventsCleared => {
                    self.window.request_redraw();
                }
                Event::RedrawRequested(_window_id) => {
                    self.draw_frame(&mut on_draw);
                }
                Event::LoopDestroyed => {
                    unsafe {
                        self.gpu
                            .device
                            .device_wait_idle()
                            .expect("Failed to wait device idle!")
                    };
                }
                _ => (),
            })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.gpu
                .device
                .destroy_command_pool(self.command_pool, None);

            self.facade.destroy(&self.gpu);
            self.apparatus.destroy(&self.gpu);

            // Uniform buffer
            self.gpu
                .device
                .destroy_descriptor_set_layout(self.uniform_buffer_layout, None);
            self.gpu
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            for buffer in self.uniform_buffers.iter() {
                buffer.destroy();
            }
            // Vertex buffer
            self.vertex_buffer.destroy();
            // Index buffer
            self.index_buffer.destroy();
            // Texture
            self.environment_texture.destroy();
            self.gpu
                .device
                .destroy_sampler(self.environment_sampler, None);
        }
    }
}

// Debug callbacks
unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

pub fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
    }
}
