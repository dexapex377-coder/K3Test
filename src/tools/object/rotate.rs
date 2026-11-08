//! Object Mode rotate: three dial rings (Blender `MAN_AXIS_ROT_*`,
//! `ED_GIZMO_DIAL_DRAW_FLAG_CLIP`, 300° arcs with gaps) plus the white
//! view-aligned trackball (`MAN_AXIS_ROT_T`).
//!
//! Angles accumulate per mouse move exactly like Blender's
//! `incremental_angle` in `transform_mode_rotate.cc`
//! (`RotationBetween` deltas + `angle_signed_on_axis`), so full turns
//! compose indefinitely instead of wrapping at +/-180 degrees.

use glam::Vec3;

use crate::scene::Scene;
use crate::tools::{segment_distance_2d, project, Axis, Editor, AXIS_HIT_RADIUS_PX};
use crate::CameraRig;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RotateKind {
    Axis(Axis),
    Trackball,
}

impl RotateKind {
    pub fn index(self) -> i32 {
        match self {
            RotateKind::Axis(axis) => axis.index() as i32,
            RotateKind::Trackball => 3,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RotateDrag {
    pub object: usize,
    pub kind: RotateKind,
    pub start_rotation: glam::Quat,
    /// Unit vector (center -> point on the gizmo sphere) at grab.
    pub grab_vec: Vec3,
    /// Unit vector from the last move event (delta tracking).
    pub last_vec: Vec3,
    /// Current unit vector; drives the sweep visual and readout.
    pub current_vec: Vec3,
    /// Total accumulated angle (radians, unbounded) for axis drags.
    pub angle: f32,
}

/// World radius of the rotate rings (Blender draws the dial around the
/// gizmo center; chosen to clear the 2 m default cube).
pub const ROTATE_RING_RADIUS: f32 = 2.2;

impl Editor {
    /// Rotate hit test: distance from the touch point to each ring
    /// (screen-space polyline, pixel tolerance like Blender).
    pub fn rotate_hit(
        &self,
        scene: &Scene,
        camera: &CameraRig,
        size: (u32, u32),
        px: f32,
        py: f32,
    ) -> Option<RotateKind> {
        let center = scene.position(self.selected?)?;
        let kinds = [
            RotateKind::Axis(Axis::X),
            RotateKind::Axis(Axis::Y),
            RotateKind::Axis(Axis::Z),
            RotateKind::Trackball,
        ];
        let mut best: Option<(RotateKind, f32)> = None;
        for kind in kinds {
            let points = ring_points(center, kind, 48);
            let mut previous = project(camera, size, points[0])?;
            let mut min_distance = f32::INFINITY;
            for point in points.iter().skip(1) {
                let Some(current) = project(camera, size, *point) else {
                    continue;
                };
                min_distance = min_distance.min(segment_distance_2d((px, py), previous, current));
                previous = current;
            }
            if min_distance <= AXIS_HIT_RADIUS_PX
                && best.is_none_or(|(_, best_d)| min_distance < best_d)
            {
                best = Some((kind, min_distance));
            }
        }
        best.map(|(kind, _)| kind)
    }

    pub fn rotate_begin(
        &mut self,
        scene: &Scene,
        camera: &CameraRig,
        size: (u32, u32),
        px: f32,
        py: f32,
    ) -> Option<RotateKind> {
        let kind = self.rotate_hit(scene, camera, size, px, py)?;
        let object = self.selected?;
        let start_rotation = scene.rotation(object)?;
        let center = scene.position(object)?;
        let (eye, dir) = Self::ray_from_screen(camera, size, px, py);
        let grab_vec = ray_sphere_point(eye, dir, center, ROTATE_RING_RADIUS)
            .map(|point| (point - center).normalize())?;
        self.rotate_dragging = Some(RotateDrag {
            object,
            kind,
            start_rotation,
            grab_vec,
            last_vec: grab_vec,
            current_vec: grab_vec,
            angle: 0.0,
        });
        Some(kind)
    }

    /// Rotate the dragged object, accumulating incremental per-event
    /// deltas so multi-turn spins compose (Blender `incremental_angle`).
    pub fn rotate_to(
        &mut self,
        scene: &mut Scene,
        camera: &CameraRig,
        size: (u32, u32),
        px: f32,
        py: f32,
    ) -> bool {
        let Some(drag) = self.rotate_dragging else {
            return false;
        };
        let center = scene.position(drag.object).unwrap_or(Vec3::ZERO);
        let (eye, dir) = Self::ray_from_screen(camera, size, px, py);
        let Some(point) = ray_sphere_point(eye, dir, center, ROTATE_RING_RADIUS) else {
            return false;
        };
        let current_vec = (point - center).normalize();
        let (rotation, angle) = match drag.kind {
            RotateKind::Axis(axis) => {
                let delta = signed_angle_deltas(drag.last_vec, current_vec, axis.direction());
                let angle = drag.angle + delta;
                let step = glam::Quat::from_axis_angle(axis.direction(), angle);
                (step * drag.start_rotation, angle)
            }
            RotateKind::Trackball => {
                let step = glam::Quat::from_rotation_arc(drag.last_vec, current_vec);
                let previous = scene.rotation(drag.object).unwrap_or(glam::Quat::IDENTITY);
                (step * previous, drag.angle)
            }
        };
        self.rotate_dragging = Some(RotateDrag {
            last_vec: current_vec,
            current_vec,
            angle,
            ..drag
        });
        let changed = scene.rotation(drag.object) != Some(rotation);
        scene.set_rotation(drag.object, rotation);
        changed
    }

    pub fn rotate_end(&mut self) {
        self.rotate_dragging = None;
    }

    pub fn rotate_degrees(&self) -> f32 {
        self.rotate_dragging
            .map(|drag| drag.angle.to_degrees())
            .unwrap_or(0.0)
    }
}

/// Signed angle from `from` to `to` around `axis`
/// (mirrors Blender's `angle_signed_on_axis_v3v3_v3` + atan2 wrap).
fn signed_angle_deltas(from: Vec3, to: Vec3, axis: Vec3) -> f32 {
    let sin = from.cross(to).dot(axis);
    let cos = from.dot(to);
    sin.atan2(cos)
}

/// Intersection of a ray with a sphere; Blender's dial grabs the ring
/// sphere itself, so this is both the hit point and the angle source.
fn ray_sphere_point(eye: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<Vec3> {
    let t = crate::tools::ray_sphere(eye, dir, center, radius)?;
    Some(eye + dir * t)
}

/// Points of a rotate ring. Axis rings span 300 deg with a gap;
/// the white trackball ring (Blender `MAN_AXIS_ROT_T` / dial FILL)
/// spans the full circle. Gaps are staggered per axis to de-overlap.
pub fn ring_points(center: Vec3, kind: RotateKind, steps: usize) -> Vec<Vec3> {
    let (axis, full) = match kind {
        RotateKind::Axis(axis) => (Some(axis), false),
        RotateKind::Trackball => (None, true),
    };
    let direction = axis.map(|a| a.direction()).unwrap_or(Vec3::Z);
    let u = if direction == Vec3::Z {
        Vec3::X
    } else {
        Vec3::Z
    };
    let v = direction.cross(u).normalize();
    let start = match kind {
        RotateKind::Axis(Axis::X) => 20.0f32.to_radians(),
        RotateKind::Axis(Axis::Y) => 140.0f32.to_radians(),
        RotateKind::Axis(Axis::Z) => 260.0f32.to_radians(),
        RotateKind::Trackball => 0.0,
    };
    let span = if full {
        std::f32::consts::TAU
    } else {
        5.236 // 300 degrees in radians
    };
    (0..=steps)
        .map(|i| {
            let angle = start + span * i as f32 / steps as f32;
            center + (u * angle.cos() + v * angle.sin()) * ROTATE_RING_RADIUS
        })
        .collect()
}
