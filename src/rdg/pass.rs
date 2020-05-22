use crate::*;

pub struct Pass<'a> {
    pub _name: String,
    pub output_textures: Vec<&'a Texture>, // The textures must live at least as long as the pass
}

impl<'a> Pass<'a> {
    pub fn new(name: &str) -> Pass {
        Pass {
            _name: String::from(name),
            output_textures: Vec::new(),
        }
    }

    pub fn with_output_texture(mut self, texture: &'a Texture) -> Pass {
        self.output_textures.push(texture);
        self
    }
}
