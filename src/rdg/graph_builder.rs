use crate::*;

#[derive(Debug, Hash)]
pub struct Pass {
    pub name: String,
    pub outputs: Vec<(vk::ImageView, vk::Format)>,
    pub input_texture: (vk::ImageView, vk::Sampler),
    pub opt_depth: Option<(vk::ImageView, vk::Format)>,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub shader_modules: Vec<vk::ShaderModule>,
    pub uniform_buffer: (vk::Buffer, usize),
}

#[derive(Hash)]
pub struct GraphBuilder {
    pub passes: Vec<Pass>,
    pub buffers: Vec<BufferHandle>,
}

impl GraphBuilder {
    pub fn new() -> GraphBuilder {
        GraphBuilder {
            passes: Vec::new(),
            buffers: Vec::new(),
        }
    }
}
