use crate::*;

pub struct RenderGraph {
    _device: ash::Device,
}

impl RenderGraph {
    pub fn new(gpu: &Gpu) -> RenderGraph {
        RenderGraph {
            _device: gpu.device.clone(),
        }
    }

    pub fn add_pass(&mut self, _render_pass: RenderPass) {}

    pub fn execute(&self) {}
}
