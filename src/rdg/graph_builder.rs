use crate::*;

#[derive(Debug, Hash)]
pub struct Pass {
    pub name: String,
    pub vertex_shader: ShaderHandle,
    pub fragment_shader: ShaderHandle,
    pub outputs: Vec<(vk::ImageView, vk::Format)>,
    pub input_texture: (vk::ImageView, vk::Sampler),
    pub opt_depth: Option<(vk::ImageView, vk::Format)>,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub uniform_buffer: (vk::Buffer, usize),
}

#[derive(Hash)]
pub struct GraphBuilder {
    pub passes: Vec<(PassHandle, Pass)>,
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
