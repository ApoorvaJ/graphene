use crate::*;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct PassHandle(u64);

#[derive(Hash)]
pub struct Pass {
    pub name: String,
    pub outputs: Vec<(vk::ImageView, vk::Format)>,
    pub input_texture: (vk::ImageView, vk::Sampler),
    pub opt_depth: Option<(vk::ImageView, vk::Format)>,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub shader_modules: Vec<vk::ShaderModule>,
    pub buffer_info: (vk::Buffer, u64), // (vk_buffer, size)
}

#[derive(Hash)]
pub struct GraphBuilder {
    pub passes: Vec<Pass>,
}

impl GraphBuilder {
    pub fn new() -> GraphBuilder {
        GraphBuilder { passes: Vec::new() }
    }

    pub fn add_pass(
        &mut self,
        name: &str,
        output_texs: &Vec<&Texture>,
        opt_depth_tex: Option<&Texture>,
        shader_modules: &Vec<vk::ShaderModule>,
        buffer: &HostVisibleBuffer,
        environment_texture: &Texture,
        environment_sampler: &Sampler,
    ) -> PassHandle {
        // TODO: Assert that color and depth textures have the same resolution
        let outputs = output_texs
            .iter()
            .map(|tex| (tex.image_view, tex.format))
            .collect();
        let opt_depth = opt_depth_tex.map(|depth_tex| (depth_tex.image_view, depth_tex.format));
        let viewport_width = output_texs[0].width;
        let viewport_height = output_texs[0].height;
        let shader_modules = shader_modules
            .iter()
            .map(|shader_module| *shader_module)
            .collect();

        self.passes.push(Pass {
            name: String::from(name),
            outputs,
            input_texture: (
                environment_texture.image_view,
                environment_sampler.vk_sampler,
            ),
            opt_depth,
            viewport_width,
            viewport_height,
            buffer_info: (buffer.vk_buffer, buffer.size),
            shader_modules,
        });

        let pass_hash: u64 = {
            let mut hasher = DefaultHasher::new();
            self.passes[self.passes.len() - 1].hash(&mut hasher);
            hasher.finish()
        };
        PassHandle(pass_hash)
    }
}
