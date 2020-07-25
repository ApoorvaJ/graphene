use crate::*;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::desktop::EventLoopExtDesktop;

const ENABLE_DEBUG_MESSENGER_CALLBACK: bool = true;

#[derive(Copy, Clone, Debug, Hash, PartialEq)]
pub struct BufferHandle(pub u64);
#[derive(Copy, Clone)]
pub struct GraphHandle(pub u64);
#[derive(Copy, Clone, Debug, Hash, PartialEq)]
pub struct PassHandle(pub u64);
#[derive(Copy, Clone)]
pub struct TextureHandle(pub u64);
#[derive(Copy, Clone, Debug, Hash, PartialEq)]
pub struct ShaderHandle(pub u64);

pub struct Context {
    window: winit::window::Window,
    event_loop: winit::event_loop::EventLoop<()>,

    pub shader_list: ShaderList,
    // TODO: Move these to the graph builder instead?
    pub texture_list: Vec<InternalTexture>,
    pub buffer_list: BufferList,

    graph_cache: Vec<(Graph, GraphHandle)>, // (graph, hash) // TODO: Make this a proper LRU and move it to its own file
    pub command_pool: vk::CommandPool,

    pub sync_idx: usize,      // Index of the synchronization primitives
    pub swapchain_idx: usize, // Index of the swapchain frame

    _watcher: notify::RecommendedWatcher, // Need to keep this alive to keep the receiver alive
    watch_rx: std::sync::mpsc::Receiver<notify::DebouncedEvent>,

    pub command_buffers: Vec<vk::CommandBuffer>,
    pub facade: Facade, // Resolution-dependent apparatus
    pub debug_utils: DebugUtils,
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

            self.facade.destroy(&mut self.texture_list);
        }
    }
}

impl Context {
    pub fn recreate_resolution_dependent_state(&mut self) {
        unsafe {
            self.gpu
                .device
                .device_wait_idle()
                .expect("Failed to wait device idle.")
        };
        // Recreate swapchain
        self.facade.destroy(&mut self.texture_list);
        self.facade = Facade::new(
            &self.basis,
            &self.gpu,
            &self.window,
            &mut self.texture_list,
            &self.debug_utils,
        );
        // Recreate the textures which depend on the resolution of the swapchain
        for i in 0..self.texture_list.len() {
            let tex = &mut self.texture_list[i];
            if let TextureKind::RelativeSized { scale } = tex.kind {
                let w = (self.facade.swapchain_width as f32 * scale) as u32;
                let h = (self.facade.swapchain_height as f32 * scale) as u32;
                tex.texture = Texture::new(
                    &tex.texture.name,
                    w,
                    h,
                    tex.texture.format,
                    tex.texture.usage,
                    tex.texture.aspect_flags,
                    &self.gpu,
                    &self.debug_utils,
                );
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
        let debug_utils = DebugUtils::new(&basis, &gpu, ENABLE_DEBUG_MESSENGER_CALLBACK);

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

        let shader_list = ShaderList::new(gpu.device.clone());

        // TODO: Move this up?
        let mut texture_list = Vec::new();
        let facade = Facade::new(&basis, &gpu, &window, &mut texture_list, &debug_utils);
        let buffer_list = BufferList::new();

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

            shader_list,
            texture_list,
            buffer_list,

            graph_cache: Vec::new(),
            command_pool,

            sync_idx: 0,
            swapchain_idx: 0,

            _watcher: watcher,
            watch_rx,

            command_buffers,
            facade,
            debug_utils,
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
                Graph::new(
                    graph_builder,
                    &self.gpu,
                    &self.shader_list,
                    &self.buffer_list,
                ),
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

        let cmd_buf = self.command_buffers[self.swapchain_idx];
        // Reset command buffer
        unsafe {
            self.gpu
                .device
                .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }
        // Begin command buffer. TODO: Is this in the right place?
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

        unsafe {
            self.gpu
                .device
                .begin_command_buffer(cmd_buf, &command_buffer_begin_info)
                .expect("Failed to begin recording command buffer.");
        }
        /* Naming the command buffer doesn't seem to work on creating it, so we
        name it on every begin frame instead.*/
        self.debug_utils
            .set_command_buffer_name(cmd_buf, &format!("command_buffer_{}", self.swapchain_idx));

        is_running
    }

    pub fn end_frame(&mut self) {
        // End command buffer. TODO: Is this in the right place?
        unsafe {
            self.gpu
                .device
                .end_command_buffer(self.command_buffers[self.swapchain_idx])
                .expect("Failed to end recording command buffer.");
        }

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
                    self.shader_list.hot_reload(&mut self.graph_cache);
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
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        output_texs: &Vec<TextureHandle>,
        opt_depth_tex: Option<TextureHandle>,
        uniform_buffer: BufferHandle,
        texture_handle: TextureHandle,
        environment_sampler: &Sampler,
    ) -> Result<PassHandle, String> {
        // TODO: Assert that color and depth textures have the same resolution
        let outputs = output_texs
            .iter()
            .map(|handle| {
                let tex = self.get_texture_from_hash(handle.0).expect(&format!(
                    "Output texture with handle `{}` not found in context.",
                    handle.0
                ));
                (tex.texture.image_view, tex.texture.format)
            })
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
            Some((depth_tex.texture.image_view, depth_tex.texture.format))
        } else {
            None
        };

        let pass = Pass {
            name: String::from(name),
            vertex_shader,
            fragment_shader,
            outputs,
            input_texture: (tex.texture.image_view, environment_sampler.vk_sampler),
            opt_depth,
            viewport_width: self.facade.swapchain_width,
            viewport_height: self.facade.swapchain_height,
            uniform_buffer,
        };

        let pass_handle = {
            let mut hasher = DefaultHasher::new();
            pass.hash(&mut hasher);
            PassHandle(hasher.finish())
        };

        graph_builder.passes.push((pass_handle, pass));

        Ok(pass_handle)
    }

    /* Shaders */
    pub fn new_shader(
        &mut self,
        name: &str,
        shader_stage: ShaderStage,
        path: &str,
    ) -> Result<ShaderHandle, String> {
        self.shader_list.new_shader(name, shader_stage, path)
    }

    /* Buffers */
    pub fn new_buffer(
        &mut self,
        name: &str,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferHandle, String> {
        self.buffer_list
            .new_buffer(name, size, usage, &self.gpu, &self.debug_utils)
    }

    pub fn upload_data<T>(&self, buffer_handle: BufferHandle, data: &[T]) {
        self.buffer_list.upload_data(buffer_handle, data);
    }
}
