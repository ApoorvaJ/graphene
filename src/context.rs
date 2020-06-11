use crate::*;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::desktop::EventLoopExtDesktop;

#[derive(Copy, Clone)]
pub struct GraphHandle(u64);
#[derive(Copy, Clone)]
pub struct TextureHandle(u64);
#[derive(Copy, Clone)]
pub struct PassHandle(u64);

#[derive(Copy, Clone)]
pub enum TextureSize {
    Absolute,                // Number of pixels
    Relative { scale: f32 }, // Scale relative to the swapchain size
}

pub struct Context {
    window: winit::window::Window,
    event_loop: winit::event_loop::EventLoop<()>,

    texture_list: Vec<(TextureHandle, Texture, TextureSize)>,
    graph_cache: Vec<(Graph, GraphHandle)>, // (graph, hash) // TODO: Make this a proper LRU and move it to its own file
    pub shader_modules: Vec<vk::ShaderModule>,
    pub command_pool: vk::CommandPool,

    pub sync_idx: usize,      // Index of the synchronization primitives
    pub swapchain_idx: usize, // Index of the swapchain frame

    _watcher: notify::RecommendedWatcher, // Need to keep this alive to keep the receiver alive
    watch_rx: std::sync::mpsc::Receiver<notify::DebouncedEvent>,

    pub command_buffers: Vec<vk::CommandBuffer>,
    pub facade: Facade, // Resolution-dependent apparatus
    pub gpu: Gpu,
    pub basis: Basis,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.gpu
                .device
                .device_wait_idle()
                .expect("Failed to wait device idle!");
            self.gpu
                .device
                .free_command_buffers(self.command_pool, &self.command_buffers);

            self.gpu
                .device
                .destroy_command_pool(self.command_pool, None);

            self.facade.destroy(&self.gpu);

            for stage in &self.shader_modules {
                self.gpu.device.destroy_shader_module(*stage, None);
            }
        }
    }
}

impl Context {
    pub fn recreate_resolution_dependent_state(&mut self) {
        unsafe {
            self.gpu
                .device
                .device_wait_idle()
                .expect("Failed to wait device idle x")
        };
        // Recreate swapchain
        self.facade.destroy(&self.gpu);
        self.facade = Facade::new(&self.basis, &self.gpu, &self.window);
        // Recreate the textures which depend on the resolution of the swapchain
        for i in 0..self.texture_list.len() {
            let (_, tex, _) = &self.texture_list[i];
            if let TextureSize::Relative { scale } = self.texture_list[i].2 {
                let w = (self.facade.swapchain_width as f32 * scale) as u32;
                let h = (self.facade.swapchain_height as f32 * scale) as u32;
                self.texture_list[i].1 =
                    Texture::new(&self.gpu, w, h, tex.format, tex.usage, tex.aspect_flags);
            }
        }
    }

    pub fn new() -> Context {
        const APP_NAME: &str = "";

        // # Init window
        let event_loop = EventLoop::new();
        let window = {
            winit::window::WindowBuilder::new()
                .with_title(APP_NAME)
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
                // .with_maximized(true)
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

        // TODO: Move this up?
        let facade = Facade::new(&basis, &gpu, &window);

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

        let (shader_modules, _) =
            utils::get_shader_modules(&gpu).expect("Failed to load shader modules");

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
            event_loop: event_loop,

            texture_list: Vec::new(),
            graph_cache: Vec::new(),
            shader_modules,
            command_pool,

            sync_idx: 0,
            swapchain_idx: 0,

            _watcher: watcher,
            watch_rx,

            command_buffers,
            facade,
            gpu,
            basis,
        }
    }

    pub fn build_graph(&mut self, graph_builder: GraphBuilder) -> GraphHandle {
        // Get the hash of the graph builder
        let req_hash: u64 = {
            let mut hasher = DefaultHasher::new();
            graph_builder.hash(&mut hasher);
            hasher.finish()
        };
        // Try finding the requested graph in the cache
        let opt_idx = self
            .graph_cache
            .iter()
            .position(|(_, cached_hash)| cached_hash.0 == req_hash);

        if opt_idx.is_none() {
            // The requested graph doesn't exist. Build it and add it to the cache.
            println!("Adding graph to cache");
            self.graph_cache.push((
                Graph::new(graph_builder, &self.gpu.device),
                GraphHandle(req_hash),
            ));
        }

        GraphHandle(req_hash)
    }

    pub fn begin_frame(&mut self) -> bool {
        let mut is_running = true;
        let mut resize_needed = false;

        let swapchain_width = self.facade.swapchain_width;
        let swapchain_height = self.facade.swapchain_height;

        self.event_loop.run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => is_running = false,
                    WindowEvent::KeyboardInput { input, .. } => match input {
                        KeyboardInput {
                            virtual_keycode,
                            state,
                            ..
                        } => match (virtual_keycode, state) {
                            (Some(VirtualKeyCode::Escape), ElementState::Pressed)
                            | (Some(VirtualKeyCode::Return), ElementState::Pressed) => {
                                is_running = false;
                            }
                            _ => {}
                        },
                    },
                    WindowEvent::Resized(physical_size) => {
                        if swapchain_width != physical_size.width
                            || swapchain_height != physical_size.height
                        {
                            resize_needed = true;
                        }
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => (),
            }
        });

        // This mechanism is need on Windows:
        if resize_needed {
            self.recreate_resolution_dependent_state();
        }

        // This mechanism suffices on Linux:
        // Acquiring the swapchain image fails if the window has been resized. If this happens, we need
        // to loop over and recreate the resolution-dependent state, and then try again.
        let mut opt_frame_idx = None;
        loop {
            let wait_fences = [self.facade.command_buffer_complete_fences[self.sync_idx]];

            unsafe {
                self.gpu
                    .device
                    .wait_for_fences(&wait_fences, true, std::u64::MAX)
                    .expect("Failed to wait for Fence.");

                let result = self.facade.ext_swapchain.acquire_next_image(
                    self.facade.swapchain,
                    std::u64::MAX,
                    self.facade.image_available_semaphores[self.sync_idx],
                    vk::Fence::null(),
                );
                match result {
                    Ok((idx, _is_suboptimal)) => {
                        opt_frame_idx = Some(idx as usize);
                    }
                    Err(error_code) => {
                        match error_code {
                            vk::Result::ERROR_OUT_OF_DATE_KHR => {
                                // Window is resized. Recreate the swapchain
                                // and exit early without drawing this frame.
                                self.recreate_resolution_dependent_state();
                            }
                            _ => panic!("Failed to acquire swapchain image."),
                        }
                    }
                }
            }

            if opt_frame_idx.is_some() {
                break;
            }
        }

        self.swapchain_idx = opt_frame_idx.unwrap();

        unsafe {
            self.gpu
                .device
                .reset_command_buffer(
                    self.command_buffers[self.swapchain_idx],
                    vk::CommandBufferResetFlags::empty(),
                )
                .unwrap();
        }

        is_running
    }

    pub fn end_frame(&mut self) {
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_semaphores = [self.facade.image_available_semaphores[self.sync_idx]];
        let signal_semaphores = [self.facade.render_finished_semaphores[self.sync_idx]];
        let command_buffers = [self.command_buffers[self.swapchain_idx as usize]];

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

        let wait_fences = [self.facade.command_buffer_complete_fences[self.sync_idx]];
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
                    self.facade.command_buffer_complete_fences[self.sync_idx],
                )
                .expect("Failed to execute queue submit.");
        }
        self.sync_idx = (self.sync_idx + 1) % self.facade.num_frames;

        let swapchains = [self.facade.swapchain];
        let image_indices = [self.swapchain_idx as u32];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        /* Present the queue */
        // According to Vulkan spec, queue_present() can fail if a resize occurs.
        // We handle this in begin_frame(), so we should be able to ignore failure here,
        // if it does happen. This works fine, when tested on Windows and on Linux on an
        // integrated GPU. If this fails on some other platform, consider calling
        // recreate_resolution_dependent_state() on error.
        let _ = unsafe {
            self.facade
                .ext_swapchain
                .queue_present(self.gpu.present_queue, &present_info)
        };

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
                    if let Some((shader_modules, _num_changed)) =
                        utils::get_shader_modules(&self.gpu)
                    {
                        // TODO: Wrap shader modules with a struct with a drop trait, and then delete this loop
                        unsafe {
                            for stage in &self.shader_modules {
                                self.gpu.device.destroy_shader_module(*stage, None);
                            }
                        }
                        self.shader_modules = shader_modules;
                    }
                }
                _ => (),
            }
        }
    }

    pub fn begin_pass(&self, graph_handle: GraphHandle, pass_handle: PassHandle) {
        let (graph, _) = self
            .graph_cache
            .iter()
            .find(|(_, cached_hash)| cached_hash.0 == graph_handle.0)
            .expect("Graph not found in cache. Have you called build_graph()?");
        graph.begin_pass(pass_handle, self.command_buffers[self.swapchain_idx])
    }

    pub fn end_pass(&self, graph_handle: GraphHandle) {
        let (graph, _) = self
            .graph_cache
            .iter()
            .find(|(_, cached_hash)| cached_hash.0 == graph_handle.0)
            .expect("Graph not found in cache. Have you called build_graph()?");
        graph.end_pass(self.command_buffers[self.swapchain_idx]);
    }

    pub fn add_pass(
        &self,
        graph_builder: &mut GraphBuilder,
        name: &str,
        output_texs: &Vec<&Texture>,
        opt_depth_tex: Option<TextureHandle>,
        shader_modules: &Vec<vk::ShaderModule>,
        buffer: &HostVisibleBuffer,
        texture_handle: TextureHandle,
        environment_sampler: &Sampler,
    ) -> Result<PassHandle, String> {
        // TODO: Assert that color and depth textures have the same resolution
        let outputs = output_texs
            .iter()
            .map(|tex| (tex.image_view, tex.format))
            .collect();
        let shader_modules = shader_modules
            .iter()
            .map(|shader_module| *shader_module)
            .collect();

        let tex = self
            .get_texture_from_hash(texture_handle.0)
            .expect(&format!(
                "Texture with handle `{}` not found in the context.",
                texture_handle.0
            ));
        let opt_depth = if let Some(depth_tex_handle) = opt_depth_tex {
            let depth_tex = self
                .get_texture_from_hash(depth_tex_handle.0)
                .expect(&format!(
                    "Depth texture with handle `{}` not found in the context.",
                    depth_tex_handle.0
                ));
            Some((depth_tex.image_view, depth_tex.format))
        } else {
            None
        };

        graph_builder.passes.push(Pass {
            name: String::from(name),
            outputs,
            input_texture: (tex.image_view, environment_sampler.vk_sampler),
            opt_depth,
            viewport_width: self.facade.swapchain_width,
            viewport_height: self.facade.swapchain_height,
            buffer_info: (buffer.vk_buffer, buffer.size),
            shader_modules,
        });

        let pass_hash: u64 = {
            let mut hasher = DefaultHasher::new();
            graph_builder.passes[graph_builder.passes.len() - 1].hash(&mut hasher);
            hasher.finish()
        };
        Ok(PassHandle(pass_hash))
    }
}

/* Textures */

impl Context {
    fn get_texture_from_hash(&self, hash: u64) -> Option<&Texture> {
        let opt_texture_and_handle = self
            .texture_list
            .iter()
            .find(|(tex_handle, _, _)| tex_handle.0 == hash);

        if let Some((_, tex, _)) = opt_texture_and_handle {
            return Some(tex);
        }

        return None;
    }

    pub fn new_texture_relative_size(
        &mut self,
        name: &str,
        scale: f32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_flags: vk::ImageAspectFlags,
    ) -> Result<TextureHandle, String> {
        // Hash texture name
        let new_hash: u64 = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        };
        // If texture with same hash already exists, return error
        if self.get_texture_from_hash(new_hash).is_some() {
            return Err(format!(
                "A texture with the same name `{}` already exists in the context.",
                name
            ));
        }
        // Create new texture
        let w = (self.facade.swapchain_width as f32 * scale) as u32;
        let h = (self.facade.swapchain_height as f32 * scale) as u32;
        let tex = Texture::new(&self.gpu, w, h, format, usage, aspect_flags);
        self.texture_list.push((
            TextureHandle(new_hash),
            tex,
            TextureSize::Relative { scale },
        ));

        Ok(TextureHandle(new_hash))
    }

    pub fn new_texture_from_file(
        &mut self,
        name: &str,
        path: &str,
    ) -> Result<TextureHandle, String> {
        // Hash texture name
        let new_hash: u64 = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        };
        // If texture with same hash already exists, return error
        if self.get_texture_from_hash(new_hash).is_some() {
            return Err(format!(
                "A texture with the same name `{}` already exists in the context.",
                name
            ));
        }
        // Create new texture
        let tex =
            Texture::new_from_image(&self.gpu, std::path::Path::new(&path), self.command_pool);
        self.texture_list
            .push((TextureHandle(new_hash), tex, TextureSize::Absolute));

        Ok(TextureHandle(new_hash))
    }
}
