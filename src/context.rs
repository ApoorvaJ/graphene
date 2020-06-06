use crate::*;

use glam::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::desktop::EventLoopExtDesktop;

#[derive(Copy, Clone)]
pub struct GraphHandle(u64);

pub struct Context {
    window: winit::window::Window,
    event_loop: winit::event_loop::EventLoop<()>,

    graph_cache: Vec<(Graph, u64)>, // (graph, hash) // TODO: Make this a proper LRU and move it to its own file
    pub shader_modules: Vec<vk::ShaderModule>,
    pub command_pool: vk::CommandPool,

    pub current_frame: usize,
    start_instant: std::time::Instant,

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
        self.facade.destroy(&self.gpu);
        self.facade = Facade::new(&self.basis, &self.gpu, &self.window);
    }

    pub fn new() -> Context {
        const APP_NAME: &str = "";

        // # Init window
        let event_loop = EventLoop::new();
        let window = {
            winit::window::WindowBuilder::new()
                .with_title(APP_NAME)
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
                .with_maximized(true)
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

            graph_cache: Vec::new(),
            shader_modules,
            command_pool,

            current_frame: 0,
            start_instant: std::time::Instant::now(),

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
            .position(|(_, cached_hash)| *cached_hash == req_hash);

        if opt_idx.is_none() {
            // The requested graph doesn't exist. Build it and add it to the cache.
            println!("Adding graph to cache");
            self.graph_cache
                .push((Graph::new(graph_builder, &self.gpu.device), req_hash));
        }

        GraphHandle(req_hash)
    }

    pub fn begin_frame(&mut self) -> (bool, usize, f32) {
        let mut is_running = true;
        let mut resize_needed = false;
        let viewport_width = self.facade.swapchain_textures[0].width;
        let viewport_height = self.facade.swapchain_textures[0].height;

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
                        if viewport_width != physical_size.width
                            || viewport_height != physical_size.height
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
            let wait_fences = [self.facade.command_buffer_complete_fences[self.current_frame]];

            unsafe {
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

        let frame_idx = opt_frame_idx.unwrap();

        let elapsed_seconds = self.start_instant.elapsed().as_secs_f32();

        unsafe {
            self.gpu
                .device
                .reset_command_buffer(
                    self.command_buffers[frame_idx],
                    vk::CommandBufferResetFlags::empty(),
                )
                .unwrap();
        }

        (is_running, frame_idx, elapsed_seconds)
    }

    pub fn end_frame(&mut self, frame_idx: usize) {
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_semaphores = [self.facade.image_available_semaphores[self.current_frame]];
        let signal_semaphores = [self.facade.render_finished_semaphores[self.current_frame]];
        let command_buffers = [self.command_buffers[frame_idx as usize]];

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

        let wait_fences = [self.facade.command_buffer_complete_fences[self.current_frame]];
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
        let image_indices = [frame_idx as u32];

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

        self.current_frame = (self.current_frame + 1) % self.facade.num_frames;
    }

    pub fn begin_pass(
        &self,
        graph_handle: GraphHandle,
        pass_handle: PassHandle,
        // TODO: Remove this param
        frame_idx: usize,
    ) {
        let (graph, _) = self
            .graph_cache
            .iter()
            .find(|(_, cached_hash)| *cached_hash == graph_handle.0)
            .expect("Graph not found in cache. Have you called build_graph()?");
        graph.begin_pass(pass_handle, self.command_buffers[frame_idx])
    }

    pub fn end_pass(
        &self,
        graph_handle: GraphHandle,
        // TODO: Remove this param
        frame_idx: usize,
    ) {
        let (graph, _) = self
            .graph_cache
            .iter()
            .find(|(_, cached_hash)| *cached_hash == graph_handle.0)
            .expect("Graph not found in cache. Have you called build_graph()?");
        graph.end_pass(self.command_buffers[frame_idx]);
    }
}
