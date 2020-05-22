// use crate::*;

pub struct Pass {
    _name: String,
    output_textures: Vec<String>,
}

impl Pass {
    pub fn new(name: &str) -> Pass {
        Pass {
            _name: String::from(name),
            output_textures: Vec::new(),
        }
    }

    pub fn with_output_texture(mut self, texture_name: &str) -> Pass {
        self.output_textures.push(String::from(texture_name));
        self
    }
}
