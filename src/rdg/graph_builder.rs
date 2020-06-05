use crate::*;

pub struct Pass {
    pub name: String,
    pub outputs: Vec<(vk::ImageView, vk::Format)>,
    pub opt_depth: Option<(vk::ImageView, vk::Format)>,
    pub viewport_width: u32,
    pub viewport_height: u32,
}

pub struct GraphBuilder {
    pub device: ash::Device,
    pub passes: Vec<Pass>,
}

impl GraphBuilder {
    pub fn new(gpu: &Gpu) -> GraphBuilder {
        GraphBuilder {
            device: gpu.device.clone(),
            passes: Vec::new(),
        }
    }

    pub fn add_pass(
        &mut self,
        name: &str,
        output_texs: &Vec<&Texture>,
        opt_depth_tex: Option<&Texture>,
    ) {
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
            opt_depth,
            viewport_width,
            viewport_height,
        });
    }
}
