use crate::*;

pub struct Sampler {
    device: ash::Device,
    pub vk_sampler: vk::Sampler,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.vk_sampler, None);
        }
    }
}

impl Sampler {
    pub fn new(gpu: &Gpu) -> Sampler {
        let vk_sampler = {
            let sampler_create_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true) // TODO: Disable this by default?
                .max_anisotropy(16.0) //
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK);

            unsafe {
                gpu.device
                    .create_sampler(&sampler_create_info, None)
                    .expect("Failed to create Sampler!")
            }
        };
        Sampler {
            device: gpu.device.clone(),
            vk_sampler,
        }
    }
}
