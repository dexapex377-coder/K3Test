//! Object Mode translate (Move): three world-axis arrows, drag
//! constrained to the grabbed axis (Blender `constraint_axis`).

use glam::Vec3;

use crate::scene::Scene;
use crate::tools::{
    segment_distance_2d, project, Axis, Editor, AXES, AXIS_HIT_RADIUS_PX,
};
use crate::CameraRig;

/// World length of a drawn axis arrow. Longer than the 2 m default cube
/// (half-size 1 m) so the manipulators are never buried inside it.
const AXIS_ARROW_LENGTH: f32 = 2.0;

#[derive(Clone, Copy, Debug)]
pub struct DragState {
    pub object: usize,
    pub axis: Axis,
    /// Axis parameter of the touch ray at grab time (prevents jumps).
    pub grab_param: f32,
    /// Object position at grab time.
    pub start_position: Vec3,
}

impl Editor {
    /// Screen-space distance test against the three axis arrows of the
    /// selected object. Returns the closest axis within tolerance.
    pub fn axis_hit(
        &self,
        scene: &Scene,
        camera: &CameraRig,
        size: (u32, u32),
        px: f32,
        py: f32,
    ) -> Option<Axis> {
        let origin = scene.position(self.selected?)?;
        let mut best: Option<(Axis, f32)> = None;
        for axis in AXES {
            let tip = origin + axis.direction() * AXIS_ARROW_LENGTH;
            let from = project(camera, size, origin)?;
            let to = project(camera, size, tip)?;
            let distance = segment_distance_2d((px, py), from, to);
            if distance <= AXIS_HIT_RADIUS_PX
                && best.is_none_or(|(_, best_d)| distance < best_d)
            {
                best = Some((axis, distance));
            }
        }
        best.map(|(axis, _)| axis)
    }

    pub fn begin_drag(
        &mut self,
        scene: &Scene,
        camera: &CameraRig,
        size: (u32, u32),
        px: f32,
        py: f32,
    ) -> Option<Axis> {
        let axis = self.axis_hit(scene, camera, size, px, py)?;
        let object = self.selected?;
        let start_position = scene.position(object)?;
        let (eye, dir) = Self::ray_from_screen(camera, size, px, py);
        let grab_param = axis_param(eye, dir, start_position, axis.direction())?;
        self.dragging = Some(DragState {
            object,
            axis,
            grab_param,
            start_position,
        });
        Some(axis)
    }

    /// Move the dragged object along its axis only (Blender's
    /// `constraint_axis` behavior). Returns true if position changed.
    pub fn drag_to(
        &mut self,
        scene: &mut Scene,
        camera: &CameraRig,
        size: (u32, u32),
        px: f32,
        py: f32,
    ) -> bool {
        let Some(drag) = self.dragging else {
            return false;
        };
        let (eye, dir) = Self::ray_from_screen(camera, size, px, py);
        let Some(param) = axis_param(eye, dir, drag.start_position, drag.axis.direction())
        else {
            return false;
        };
        let delta = param - drag.grab_param;
        let new_position = drag.start_position + drag.axis.direction() * delta;
        let changed = scene.position(drag.object) != Some(new_position);
        scene.set_position(drag.object, new_position);
        changed
    }

    pub fn end_drag(&mut self) {
        self.dragging = None;
    }
}

/// Closest-point parameter between a ray and a world axis line through
/// `origin` along `direction` — the exact same ray-ray closest-point
/// computation Blender uses via `isect_ray_ray_v3` in
/// `transform_constraints.cc` for axis-constrained translate.
/// None when the ray is (near) parallel to the axis, which freezes the
/// drag instead of producing jumps.
fn axis_param(ray_origin: Vec3, ray_dir: Vec3, origin: Vec3, direction: Vec3) -> Option<f32> {
    let w0 = ray_origin - origin;
    let b = ray_dir.dot(direction);
    let d = ray_dir.dot(w0);
    let e = direction.dot(w0);
    let denom = 1.0 - b * b;
    if denom.abs() < 0.02 {
        return None;
    }
    Some((e - b * d) / denom)
}

/// Solid arrow geometry (triangles) for the move gizmo: a hexagonal
/// shaft plus a conical head per axis. Solid triangles are needed
/// because Vulkan/WGPU lines are always 1 px wide; triangles give the
/// manipulators real thickness from every angle, like Blender's gizmo
/// meshes.
pub fn build_gizmo_vertices(origin: Vec3) -> Vec<crate::Vertex> {
    let mut out = Vec::with_capacity(162);
    let colors = [
        glam::Vec3::new(1.0, 0.25, 0.25),
        glam::Vec3::new(0.3, 1.0, 0.3),
        glam::Vec3::new(0.3, 0.55, 1.0),
    ];
    let shaft_radius = 0.065;
    let head_radius = 0.17;
    let head_length = 0.42;
    let shaft_end = AXIS_ARROW_LENGTH - head_length;
    const SIDES: usize = 6;

    for (axis, color) in AXES.iter().zip(colors.iter()) {
        let direction = axis.direction();
        let u = if *axis == Axis::Z {
            glam::Vec3::X
        } else {
            glam::Vec3::Z
        };
        let v = direction.cross(u).normalize();
        let ring = |center: Vec3, radius: f32| {
            let mut points = [Vec3::ZERO; SIDES];
            for (index, slot) in points.iter_mut().enumerate() {
                let angle = index as f32 * std::f32::consts::TAU / SIDES as f32;
                *slot = center + (u * angle.cos() + v * angle.sin()) * radius;
            }
            points
        };
        let base = ring(origin, shaft_radius);
        let mid = ring(origin + direction * shaft_end, shaft_radius);
        let head_base = ring(origin + direction * shaft_end, head_radius);
        let tip = origin + direction * AXIS_ARROW_LENGTH;
        for side in 0..SIDES {
            let next = (side + 1) % SIDES;
            // Shaft quad (2 triangles).
            for tri in [
                [base[side], base[next], mid[next]],
                [base[side], mid[next], mid[side]],
            ] {
                for point in tri {
                    out.push(crate::Vertex::new(point, *color));
                }
            }
            // Cone head side (1 triangle).
            for point in [tip, head_base[side], head_base[next]] {
                out.push(crate::Vertex::new(point, *color));
            }
        }
    }
    out
}
