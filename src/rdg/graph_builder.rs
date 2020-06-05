use crate::*;

pub struct Pass {
    pub name: String,
    pub outputs: Vec<(vk::ImageView, vk::Format)>,
    pub input_texture: (vk::ImageView, vk::Sampler),
    pub opt_depth: Option<(vk::ImageView, vk::Format)>,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub buffer_info: (vk::Buffer, u64), // (vk_buffer, size)
}

pub struct GraphBuilder {
    pub device: ash::Device,
    pub passes: Vec<Pass>,
    next_pass_handle: u64, // TODO: When we switch to pass hashing, delete this.
}

impl GraphBuilder {
    pub fn new(gpu: &Gpu) -> GraphBuilder {
        GraphBuilder {
            device: gpu.device.clone(),
            passes: Vec::new(),
            next_pass_handle: 0,
        }
    }

    pub fn add_pass(
        &mut self,
        name: &str,
        output_texs: &Vec<&Texture>,
        opt_depth_tex: Option<&Texture>,
        // TODO: Remove these params
        buffer: &HostVisibleBuffer,
        environment_texture: &Texture,
        environment_sampler: vk::Sampler,
    ) -> u64 {
        // TODO: Assert that color and depth textures have the same resolution
        let outputs = output_texs
            .iter()
            .map(|tex| (tex.image_view, tex.format))
            .collect();
        let opt_depth = opt_depth_tex.map(|depth_tex| (depth_tex.image_view, depth_tex.format));
        let viewport_width = output_texs[0].width;
        let viewport_height = output_texs[0].height;

        self.passes.push(Pass {
            name: String::from(name),
            outputs,
            input_texture: (environment_texture.image_view, environment_sampler),
            opt_depth,
            viewport_width,
            viewport_height,
            buffer_info: (buffer.vk_buffer, buffer.size),
        });

        let pass_handle = self.next_pass_handle;
        self.next_pass_handle += 1;
        pass_handle
    }
}
