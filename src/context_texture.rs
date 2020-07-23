use crate::*;

#[derive(Copy, Clone, PartialEq)]
pub enum TextureKind {
    Swapchain,
    AbsoluteSized,
    RelativeSized { scale: f32 }, // Scale relative to the swapchain size
}

pub struct InternalTexture {
    pub handle: TextureHandle,
    pub texture: Texture,
    pub kind: TextureKind,
}

impl Context {
    pub fn get_texture_from_hash(&self, hash: u64) -> Option<&InternalTexture> {
        self.texture_list.iter().find(|tex| tex.handle.0 == hash)
    }

    pub fn new_texture_relative_size(
        &mut self,
        name: &str,
        scale: f32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_flags: vk::ImageAspectFlags,
    ) -> Result<TextureHandle, String> {
        // Hash texture name
        let new_hash: u64 = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        };
        // If texture with same hash already exists, return error
        if self.get_texture_from_hash(new_hash).is_some() {
            return Err(format!(
                "A texture with the same name `{}` already exists in the context.",
                name
            ));
        }
        // Create new texture
        let w = (self.facade.swapchain_width as f32 * scale) as u32;
        let h = (self.facade.swapchain_height as f32 * scale) as u32;
        let tex = Texture::new(
            &self.gpu,
            w,
            h,
            format,
            usage,
            aspect_flags,
            name,
            &self.debug_utils,
        );
        self.texture_list.push(InternalTexture {
            handle: TextureHandle(new_hash),
            texture: tex,
            kind: TextureKind::RelativeSized { scale },
        });

        Ok(TextureHandle(new_hash))
    }

    pub fn new_texture_from_file(
        &mut self,
        name: &str,
        path: &str,
    ) -> Result<TextureHandle, String> {
        // Hash texture name
        let new_hash: u64 = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        };
        // If texture with same hash already exists, return error
        if self.get_texture_from_hash(new_hash).is_some() {
            return Err(format!(
                "A texture with the same name `{}` already exists in the context.",
                name
            ));
        }
        // Create new texture
        let tex = Texture::new_from_image(
            &self.gpu,
            std::path::Path::new(&path),
            self.command_pool,
            name,
            &self.debug_utils,
        );
        self.texture_list.push(InternalTexture {
            handle: TextureHandle(new_hash),
            texture: tex,
            kind: TextureKind::AbsoluteSized,
        });

        Ok(TextureHandle(new_hash))
    }
}
