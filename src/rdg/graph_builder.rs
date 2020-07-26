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
