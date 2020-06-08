use crate::*;

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
}
