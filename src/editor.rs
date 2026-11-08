//! k3-editor (module): tools, selection and translate-gizmo math.
//!
//! Ported behavior from Blender's `transform_gizmo_3d.cc`
//! (`gizmogroup_init_properties_from_twtype` binds each axis manipulator
//! to a translate restricted to one world axis) and `view3d_select.cc`
//! (`ED_view3d_select_dist_px`: pixel-tolerance hit testing). No desktop
//! architecture is copied: single selection, touch-first picking.

use glam::Vec3;

use crate::scene::Scene;
use crate::CameraRig;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tool {
    None,
    Move,
}

impl Tool {
    pub fn name(self) -> &'static str {
        match self {
            Tool::None => "none",
            Tool::Move => "move",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }

    pub fn direction(self) -> Vec3 {
        match self {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }
}

pub const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

/// Screen-space hit tolerance in pixels for axis manipulators.
/// Deliberately larger than the 1 px visual line width: Blender keeps
/// manipulators grabbable on touch screens (`ED_view3d_select_dist_px`
/// is a pixel-space tolerance, not a world-space one).
const AXIS_HIT_RADIUS_PX: f32 = 72.0;

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

#[derive(Clone)]
pub struct Editor {
    pub tool: Tool,
    /// Single selection; the selected object is also the active object.
    pub selected: Option<usize>,
    pub dragging: Option<DragState>,
}

impl Editor {
    pub fn new() -> Self {
        Self {
            tool: Tool::None,
            selected: None,
            dragging: None,
        }
    }

    pub fn show_gizmo(&self) -> bool {
        self.tool == Tool::Move && self.selected.is_some()
    }

    /// Touch point (device pixels) -> world ray (origin = eye).
    pub fn ray_from_screen(
        camera: &CameraRig,
        size: (u32, u32),
        px: f32,
        py: f32,
    ) -> (Vec3, Vec3) {
        let eye = camera.eye();
        let forward = (camera.target - eye).normalize();
        let right = forward.cross(crate::WORLD_UP).normalize();
        let up = right.cross(forward).normalize();
        let width = size.0.max(1) as f32;
        let height = size.1.max(1) as f32;
        let aspect = width / height;
        let focal = 1.0 / (crate::VIEW_FOV_Y * 0.5).tan();
        let ndc_x = (px / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / height) * 2.0;
        let dir = (forward + right * (ndc_x * aspect / focal)
            + up * (ndc_y / focal))
        .normalize();
        (eye, dir)
    }

    /// Nearest object hit along the ray (AABB for the cube, proxy sphere
    /// for the camera/light icons). Returns (object index, hit distance).
    pub fn pick(scene: &Scene, eye: Vec3, dir: Vec3) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (index, object) in scene.objects.iter().enumerate() {
            let hit = match object.kind {
                crate::scene::ObjectKind::Cube { half_size } => {
                    ray_aabb(eye, dir, object.position, half_size)
                }
                _ => ray_sphere(eye, dir, object.position, scene.pick_radius(index)),
            };
            if let Some(t) = hit {
                if best.is_none_or(|(_, best_t)| t < best_t) {
                    best = Some((index, t));
                }
            }
        }
        best
    }

    /// Tap selection: hit -> select (and make active), miss -> clear.
    pub fn apply_tap_pick(&mut self, scene: &Scene, eye: Vec3, dir: Vec3) {
        self.dragging = None;
        self.selected = Self::pick(scene, eye, dir).map(|(index, _)| index);
    }

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
    pub fn drag_to(&mut self, scene: &mut Scene, camera: &CameraRig, size: (u32, u32), px: f32, py: f32) -> bool {
        let Some(drag) = self.dragging else {
            return false;
        };
        let (eye, dir) = Self::ray_from_screen(camera, size, px, py);
        let Some(param) = axis_param(eye, dir, drag.start_position, drag.axis.direction()) else {
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

fn ray_aabb(eye: Vec3, dir: Vec3, center: Vec3, half: f32) -> Option<f32> {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    let origins = [eye.x, eye.y, eye.z];
    let dirs = [dir.x, dir.y, dir.z];
    let centers = [center.x, center.y, center.z];
    for i in 0..3 {
        if dirs[i].abs() < 1e-8 {
            if origins[i] < centers[i] - half || origins[i] > centers[i] + half {
                return None;
            }
        } else {
            let mut t0 = (centers[i] - half - origins[i]) / dirs[i];
            let mut t1 = (centers[i] + half - origins[i]) / dirs[i];
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
            if t_min > t_max {
                return None;
            }
        }
    }
    if t_max < 0.0 {
        return None;
    }
    Some(if t_min >= 0.0 { t_min } else { 0.0 })
}

fn ray_sphere(eye: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = eye - center;
    let b = oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t >= 0.0 {
        Some(t)
    } else {
        None
    }
}

fn project(camera: &CameraRig, size: (u32, u32), world: Vec3) -> Option<(f32, f32)> {
    let view = camera.view_matrix();
    let projection = glam::Mat4::perspective_rh(
        crate::VIEW_FOV_Y,
        size.0.max(1) as f32 / size.1.max(1) as f32,
        crate::VIEW_NEAR,
        crate::VIEW_FAR,
    );
    let clip = projection
        * view
        * glam::Vec4::new(world.x, world.y, world.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = glam::Vec3::new(clip.x, clip.y, clip.z) / clip.w;
    Some((
        (ndc.x * 0.5 + 0.5) * size.0 as f32,
        (0.5 - ndc.y * 0.5) * size.1 as f32,
    ))
}

fn segment_distance_2d(point: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let ab = (b.0 - a.0, b.1 - a.1);
    let len_sq = ab.0 * ab.0 + ab.1 * ab.1;
    let t = if len_sq > 0.0 {
        (((point.0 - a.0) * ab.0 + (point.1 - a.1) * ab.1) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = (a.0 + ab.0 * t, a.1 + ab.1 * t);
    ((point.0 - closest.0).powi(2) + (point.1 - closest.1).powi(2)).sqrt()
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
    let shaft_end = crate::editor::AXIS_ARROW_LENGTH - head_length;
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
        let tip = origin + direction * crate::editor::AXIS_ARROW_LENGTH;
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
