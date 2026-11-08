//! k3-scene (module): real object state for the default Blender scene.
//!
//! Transformations live here, not in the renderer: the renderer only
//! receives a snapshot of positions and draws them.

use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObjectKind {
    Cube { half_size: f32 },
    Camera,
    Light,
}

#[derive(Clone, Copy, Debug)]
pub struct SceneObject {
    pub kind: ObjectKind,
    pub position: Vec3,
}

#[derive(Clone)]
pub struct Scene {
    pub objects: Vec<SceneObject>,
}

pub const SCENE_CUBE: usize = 0;
pub const SCENE_CAMERA: usize = 1;
pub const SCENE_LIGHT: usize = 2;

impl Scene {
    pub fn default_scene(
        camera_position: Vec3,
        light_position: Vec3,
    ) -> Self {
        Self {
            objects: vec![
                SceneObject {
                    kind: ObjectKind::Cube { half_size: 1.0 },
                    position: Vec3::ZERO,
                },
                SceneObject {
                    kind: ObjectKind::Camera,
                    position: camera_position,
                },
                SceneObject {
                    kind: ObjectKind::Light,
                    position: light_position,
                },
            ],
        }
    }

    pub fn position(&self, index: usize) -> Option<Vec3> {
        self.objects.get(index).map(|object| object.position)
    }

    pub fn set_position(&mut self, index: usize, position: Vec3) {
        if let Some(object) = self.objects.get_mut(index) {
            object.position = position;
        }
    }

    /// Picking proxy radius used for the camera/light icons.
    pub fn pick_radius(&self, index: usize) -> f32 {
        match self.objects.get(index).map(|object| object.kind) {
            Some(ObjectKind::Cube { half_size }) => half_size * 1.8,
            Some(ObjectKind::Camera) => 0.75,
            Some(ObjectKind::Light) => 0.85,
            None => 0.0,
        }
    }
}
