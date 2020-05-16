// use crate::*;

pub struct RenderPass {
    _name: String,
    output_textures: Vec<String>,
}

impl RenderPass {
    pub fn new(name: &str) -> RenderPass {
        RenderPass {
            _name: String::from(name),
            output_textures: Vec::new(),
        }
    }

    pub fn with_output_texture(mut self, texture_name: &str) -> RenderPass {
        self.output_textures.push(String::from(texture_name));
        self
    }
}
