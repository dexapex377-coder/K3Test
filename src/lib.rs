//! K3Test Phase 1.0 — Android-first wgpu viewport.
//!
//! A small DCC-shaped core, not a desktop Blender port: Kotlin owns the Activity,
//! SurfaceView and touch gestures; the Rust worker owns scene snapshot, camera,
//! ANativeWindow, wgpu surface and GPU resources.

//! The ANativeWindow ownership contract is deliberate:
//! * `ANativeWindow_fromSurface` acquires one native reference.
//! * The worker owns that reference for the lifetime of its wgpu surface.
//! * On shutdown the wgpu surface is dropped before the retained
//!   `NativeWindow`, so `ANativeWindow_release` happens exactly once.
//! * Kotlin teardown is non-blocking; the worker performs every GPU drop on
//!   its own render thread.

use std::{
    collections::{HashMap, VecDeque},
    ffi::c_void,
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Condvar, Mutex, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use jni::{
    objects::JObject,
    sys::{jlong, jstring},
    JNIEnv,
};
use raw_window_handle::{
    AndroidDisplayHandle, AndroidNdkWindowHandle, DisplayHandle, HandleError, HasDisplayHandle,
    HasWindowHandle, RawDisplayHandle, RawWindowHandle, WindowHandle,
};
use wgpu::util::DeviceExt;

pub mod editor;
pub mod scene;

use editor::{Editor, Tool};
use scene::Scene;

const VERTEX_SHADER: &str = r"
@group(0) @binding(0) var<uniform> vp: mat4x4<f32>;
@group(0) @binding(1) var<uniform> light_center: vec4<f32>;
@group(0) @binding(2) var<uniform> light_right: vec4<f32>;
@group(0) @binding(3) var<uniform> light_up: vec4<f32>;
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vp * vec4<f32>(input.position, 1.0);
    out.color = input.color;
    return out;
}
// Screen-aligned point-light icon, Blender-style: the geometry lives in a
// local XY plane and is billboarded around `light_center` with the view
// right/up axes, so it always stands facing the viewer.
@vertex
fn vs_light(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = light_center.xyz
        + light_right.xyz * input.position.x
        + light_up.xyz * input.position.y;
    out.clip_position = vp * vec4<f32>(world, 1.0);
    out.color = input.color;
    return out;
}
";

const FRAGMENT_SHADER: &str = r"
@fragment
fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}
";

pub(crate) const DEFAULT_CAMERA_POSITION: glam::Vec3 =
    glam::Vec3::new(7.358891, -6.925791, 4.958309);
pub(crate) const DEFAULT_LIGHT_POSITION: glam::Vec3 =
    glam::Vec3::new(4.076245, 1.005454, 5.903862);
pub(crate) const WORLD_UP: glam::Vec3 = glam::Vec3::Z;
pub(crate) const VIEW_FOV_Y: f32 = 50.0 * std::f32::consts::PI / 180.0;
pub(crate) const VIEW_NEAR: f32 = 0.1;
pub(crate) const VIEW_FAR: f32 = 100.0;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    pub(crate) fn new(position: glam::Vec3, color: glam::Vec3) -> Self {
        Self {
            position: position.to_array(),
            color: color.to_array(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraRig {
    pub(crate) target: glam::Vec3,
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) distance: f32,
}

impl Default for CameraRig {
    fn default() -> Self {
        let position = DEFAULT_CAMERA_POSITION;
        let distance = position.length();
        Self {
            target: glam::Vec3::ZERO,
            yaw: position.y.atan2(position.x),
            pitch: position.z.atan2((position.x * position.x + position.y * position.y).sqrt()),
            distance,
        }
    }
}

impl CameraRig {
    pub(crate) fn eye(&self) -> glam::Vec3 {
        let horizontal = self.pitch.cos();
        let offset = glam::Vec3::new(
            horizontal * self.yaw.cos(),
            horizontal * self.yaw.sin(),
            self.pitch.sin(),
        ) * self.distance;
        self.target + offset
    }

    pub(crate) fn view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.eye(), self.target, WORLD_UP)
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        // Blender-style damped orbit: ~0.003 rad/px keeps a full-screen drag
        // near ~180 degrees instead of spinning multiple turns.
        self.yaw -= dx * 0.003;
        self.pitch = (self.pitch + dy * 0.003).clamp(-1.45, 1.45);
    }

    fn pan(&mut self, dx: f32, dy: f32, size: (u32, u32)) {
        let eye = self.eye();
        let forward = (self.target - eye).normalize();
        let right = forward.cross(WORLD_UP).normalize();
        let up = right.cross(forward).normalize();
        let pixels = size.0.min(size.1).max(1) as f32;
        let world_per_pixel =
            (2.0 * self.distance * (VIEW_FOV_Y * 0.5).tan()) / pixels;
        self.target += right * (-dx * world_per_pixel) + up * (dy * world_per_pixel);
    }

    fn zoom(&mut self, scale: f32) {
        if scale > 0.0 {
            // Keep the eye outside the 2 m default cube (bounding sphere
            // ~1.74 m) so pinch-zoom can never end up inside it.
            self.distance = (self.distance * scale).clamp(3.0, 60.0);
        }
    }

    fn frame_all(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy)]
enum CameraCommand {
    Orbit { dx: f32, dy: f32 },
    Pan { dx: f32, dy: f32 },
    Zoom { scale: f32 },
    FrameAll,
}

impl CameraRig {
    fn apply(&mut self, command: CameraCommand, size: (u32, u32)) {
        match command {
            CameraCommand::Orbit { dx, dy } => self.orbit(dx, dy),
            CameraCommand::Pan { dx, dy } => self.pan(dx, dy, size),
            CameraCommand::Zoom { scale } => self.zoom(scale),
            CameraCommand::FrameAll => self.frame_all(),
        }
    }
}

// Projection/view math comes from glam (standard right-handed look_at with
// [0,1] depth for Vulkan/WebGPU), matching Blender's Z-up orbit model.
// Do not hand-roll these again: the previous custom matrix had both depth
// signs flipped, which clipped every fragment and left a flat clear color.

pub(crate) fn push_line(out: &mut Vec<Vertex>, a: glam::Vec3, b: glam::Vec3, color: glam::Vec3) {
    out.push(Vertex::new(a, color));
    out.push(Vertex::new(b, color));
}

fn push_box_edges(
    out: &mut Vec<Vertex>,
    center: glam::Vec3,
    right: glam::Vec3,
    up: glam::Vec3,
    half_right: f32,
    half_up: f32,
    color: glam::Vec3,
) {
    let corners = [
        center - right * half_right - up * half_up,
        center + right * half_right - up * half_up,
        center + right * half_right + up * half_up,
        center - right * half_right + up * half_up,
    ];
    for i in 0..4 {
        push_line(out, corners[i], corners[(i + 1) % 4], color);
        let other_center = center + right * half_right * 0.0;
        let _ = other_center;
    }
}

fn build_cube_vertices(offset: glam::Vec3, rotation: glam::Quat) -> [Vertex; 36] {
    let face = |color: glam::Vec3| color;
    let g = 0.72;
    let gray = |shade: f32| face(glam::Vec3::splat(g * shade));
    let mut vertices = [
        // front/back/right/left/top/bottom, each two triangles
        Vertex::new(glam::Vec3::new(-1.0, -1.0, 1.0), gray(1.0)),
        Vertex::new(glam::Vec3::new(1.0, -1.0, 1.0), gray(1.0)),
        Vertex::new(glam::Vec3::new(1.0, 1.0, 1.0), gray(1.0)),
        Vertex::new(glam::Vec3::new(-1.0, -1.0, 1.0), gray(1.0)),
        Vertex::new(glam::Vec3::new(1.0, 1.0, 1.0), gray(1.0)),
        Vertex::new(glam::Vec3::new(-1.0, 1.0, 1.0), gray(1.0)),
        Vertex::new(glam::Vec3::new(-1.0, -1.0, -1.0), gray(0.42)),
        Vertex::new(glam::Vec3::new(-1.0, 1.0, -1.0), gray(0.42)),
        Vertex::new(glam::Vec3::new(1.0, 1.0, -1.0), gray(0.42)),
        Vertex::new(glam::Vec3::new(-1.0, -1.0, -1.0), gray(0.42)),
        Vertex::new(glam::Vec3::new(1.0, 1.0, -1.0), gray(0.42)),
        Vertex::new(glam::Vec3::new(1.0, -1.0, -1.0), gray(0.42)),
        Vertex::new(glam::Vec3::new(1.0, -1.0, -1.0), gray(0.62)),
        Vertex::new(glam::Vec3::new(1.0, 1.0, -1.0), gray(0.62)),
        Vertex::new(glam::Vec3::new(1.0, 1.0, 1.0), gray(0.62)),
        Vertex::new(glam::Vec3::new(1.0, -1.0, -1.0), gray(0.62)),
        Vertex::new(glam::Vec3::new(1.0, 1.0, 1.0), gray(0.62)),
        Vertex::new(glam::Vec3::new(1.0, -1.0, 1.0), gray(0.62)),
        Vertex::new(glam::Vec3::new(-1.0, -1.0, -1.0), gray(0.78)),
        Vertex::new(glam::Vec3::new(-1.0, -1.0, 1.0), gray(0.78)),
        Vertex::new(glam::Vec3::new(-1.0, 1.0, 1.0), gray(0.78)),
        Vertex::new(glam::Vec3::new(-1.0, -1.0, -1.0), gray(0.78)),
        Vertex::new(glam::Vec3::new(-1.0, 1.0, 1.0), gray(0.78)),
        Vertex::new(glam::Vec3::new(-1.0, 1.0, -1.0), gray(0.78)),
        Vertex::new(glam::Vec3::new(-1.0, 1.0, -1.0), gray(1.12)),
        Vertex::new(glam::Vec3::new(-1.0, 1.0, 1.0), gray(1.12)),
        Vertex::new(glam::Vec3::new(1.0, 1.0, 1.0), gray(1.12)),
        Vertex::new(glam::Vec3::new(-1.0, 1.0, -1.0), gray(1.12)),
        Vertex::new(glam::Vec3::new(1.0, 1.0, 1.0), gray(1.12)),
        Vertex::new(glam::Vec3::new(1.0, 1.0, -1.0), gray(1.12)),
        Vertex::new(glam::Vec3::new(-1.0, -1.0, -1.0), gray(0.32)),
        Vertex::new(glam::Vec3::new(1.0, -1.0, -1.0), gray(0.32)),
        Vertex::new(glam::Vec3::new(1.0, -1.0, 1.0), gray(0.32)),
        Vertex::new(glam::Vec3::new(-1.0, -1.0, -1.0), gray(0.32)),
        Vertex::new(glam::Vec3::new(1.0, -1.0, 1.0), gray(0.32)),
        Vertex::new(glam::Vec3::new(-1.0, -1.0, 1.0), gray(0.32)),
    ];
    for vertex in &mut vertices {
        let local = glam::Vec3::new(vertex.position[0], vertex.position[1], vertex.position[2]);
        let world = offset + rotation * local;
        vertex.position = [world.x, world.y, world.z];
    }
    vertices
}

fn build_cube_edges(offset: glam::Vec3, rotation: glam::Quat, color: glam::Vec3) -> Vec<Vertex> {
    let corner = |x: f32, y: f32, z: f32| {
        offset + rotation * glam::Vec3::new(x, y, z)
    };
    let edges = [
        (corner(-1.0, -1.0, -1.0), corner(1.0, -1.0, -1.0)),
        (corner(1.0, -1.0, -1.0), corner(1.0, 1.0, -1.0)),
        (corner(1.0, 1.0, -1.0), corner(-1.0, 1.0, -1.0)),
        (corner(-1.0, 1.0, -1.0), corner(-1.0, -1.0, -1.0)),
        (corner(-1.0, -1.0, 1.0), corner(1.0, -1.0, 1.0)),
        (corner(1.0, -1.0, 1.0), corner(1.0, 1.0, 1.0)),
        (corner(1.0, 1.0, 1.0), corner(-1.0, 1.0, 1.0)),
        (corner(-1.0, 1.0, 1.0), corner(-1.0, -1.0, 1.0)),
        (corner(-1.0, -1.0, -1.0), corner(-1.0, -1.0, 1.0)),
        (corner(1.0, -1.0, -1.0), corner(1.0, -1.0, 1.0)),
        (corner(1.0, 1.0, -1.0), corner(1.0, 1.0, 1.0)),
        (corner(-1.0, 1.0, -1.0), corner(-1.0, 1.0, 1.0)),
    ];
    let mut out = Vec::with_capacity(24);
    for (a, b) in edges {
        push_line(&mut out, a, b, color);
    }
    out
}

fn build_grid_vertices() -> Vec<Vertex> {
    let mut out = Vec::with_capacity(84);
    let gray = glam::Vec3::splat(0.22);
    let major = glam::Vec3::splat(0.34);
    let x_axis = glam::Vec3::new(0.8, 0.2, 0.2);
    let y_axis = glam::Vec3::new(0.2, 0.8, 0.2);
    for i in -10..=10 {
        let coordinate = i as f32;
        let x_color = if i == 0 {
            x_axis
        } else if i % 5 == 0 {
            major
        } else {
            gray
        };
        let y_color = if i == 0 {
            y_axis
        } else if i % 5 == 0 {
            major
        } else {
            gray
        };
        push_line(
            &mut out,
            glam::Vec3::new(coordinate, -10.0, 0.0),
            glam::Vec3::new(coordinate, 10.0, 0.0),
            x_color,
        );
        push_line(
            &mut out,
            glam::Vec3::new(-10.0, coordinate, 0.0),
            glam::Vec3::new(10.0, coordinate, 0.0),
            y_color,
        );
    }
    out
}

fn build_camera_glyph_vertices(
    center: glam::Vec3,
    rotation: glam::Quat,
    color: glam::Vec3,
) -> Vec<Vertex> {
    // Blender-style camera icon: a recognizable rectangular body behind the
    // lens plus the field-of-view frustum in front. The icon follows the
    // object's live position and rotation (default aim toward the origin).
    let position = center;
    let forward = rotation * (glam::Vec3::ZERO - DEFAULT_CAMERA_POSITION).normalize();
    let right = forward.cross(WORLD_UP).normalize();
    let up = right.cross(forward).normalize();

    let rect = |center: glam::Vec3, half_right: f32, half_up: f32| {
        [
            center - right * half_right - up * half_up,
            center + right * half_right - up * half_up,
            center + right * half_right + up * half_up,
            center - right * half_right + up * half_up,
        ]
    };

    // Body: 0.55 x 0.42 section, 0.80 long, lens at `position`.
    let front = rect(position, 0.275, 0.21);
    let back = rect(position - forward * 0.80, 0.275, 0.21);
    // Frustum: Blender view-frame projection for the default 50 mm lens
    // over a 36 mm sensor, drawn at a 2.5 m draw distance so the icon
    // stays proportional to the 2 m default cube.
    let far = rect(position + forward * 2.5, 0.90, 0.51);

    let mut out = Vec::with_capacity(76);
    for i in 0..4 {
        // Body outline.
        push_line(&mut out, front[i], front[(i + 1) % 4], color);
        push_line(&mut out, back[i], back[(i + 1) % 4], color);
        push_line(&mut out, front[i], back[i], color);
        // Frustum: spokes from the lens face + far rectangle.
        push_line(&mut out, front[i], far[i], color);
        push_line(&mut out, far[i], far[(i + 1) % 4], color);
    }
    // Lens glass: inset rectangle on the front face.
    let lens = rect(position, 0.16, 0.12);
    for i in 0..4 {
        push_line(&mut out, lens[i], lens[(i + 1) % 4], color);
    }
    // Viewfinder hump on top of the body.
    let hump_front = rect(position - forward * 0.15 + up * 0.21, 0.07, 0.04);
    let hump_back = rect(position - forward * 0.45 + up * 0.21, 0.07, 0.04);
    for i in 0..4 {
        push_line(&mut out, hump_front[i], hump_front[(i + 1) % 4], color);
        push_line(&mut out, hump_back[i], hump_back[(i + 1) % 4], color);
        push_line(&mut out, hump_front[i], hump_back[i], color);
    }
    // Blender camera_tria: up marker attached to the body front face
    // (base is the front top edge, apex is the lens center).
    push_line(&mut out, position, front[2], color);
    push_line(&mut out, position, front[3], color);
    out
}

fn build_light_glyph_vertices(color: glam::Vec3) -> Vec<Vertex> {
    // Screen-aligned point-light icon (Blender light_point_lines style):
    // a small ring + center diamond + 8 symmetric rays, authored in a
    // local XY plane around the origin and billboarded by `vs_light`
    // around the scene light position. Lines only, built on scene sync.
    let ring_radius = 0.30;
    let diamond_radius = 0.12;
    let ray_inner = 0.42;
    let ray_outer = 0.75;
    const segments: usize = 16;
    let mut ring = [glam::Vec3::ZERO; segments];
    for (index, slot) in ring.iter_mut().enumerate() {
        let angle = index as f32 * std::f32::consts::TAU / segments as f32;
        *slot = glam::Vec3::new(angle.cos(), angle.sin(), 0.0) * ring_radius;
    }
    let mut diamond = [glam::Vec3::ZERO; 4];
    for (index, slot) in diamond.iter_mut().enumerate() {
        let angle = index as f32 * std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_4;
        *slot = glam::Vec3::new(angle.cos(), angle.sin(), 0.0) * diamond_radius;
    }
    let mut out = Vec::with_capacity(56);
    for index in 0..segments {
        push_line(&mut out, ring[index], ring[(index + 1) % segments], color);
    }
    for index in 0..4 {
        push_line(&mut out, diamond[index], diamond[(index + 1) % 4], color);
    }
    for index in 0..8 {
        let angle = index as f32 * std::f32::consts::FRAC_PI_4;
        let direction = glam::Vec3::new(angle.cos(), angle.sin(), 0.0);
        push_line(
            &mut out,
            direction * ray_inner,
            direction * ray_outer,
            color,
        );
    }
    out
}

/// All world-space overlay lines: floor grid, camera glyph, move gizmo
/// and selection outline. Rebuilt only when the scene/editor state
/// changes (never per frame while idle).
fn build_world_lines(scene: &Scene, editor: &Editor) -> Vec<Vertex> {
    let mut out = build_grid_vertices();
    let camera_selected = editor.selected == Some(scene::SCENE_CAMERA);
    let camera_color = if camera_selected {
        glam::Vec3::new(1.0, 0.6, 0.1)
    } else {
        glam::Vec3::new(0.45, 0.9, 1.0)
    };
    if let Some(position) = scene.position(scene::SCENE_CAMERA) {
        let rotation = scene
            .rotation(scene::SCENE_CAMERA)
            .unwrap_or(glam::Quat::IDENTITY);
        out.extend(build_camera_glyph_vertices(position, rotation, camera_color));
    }
    if editor.tool == Tool::Rotate {
        if let Some(center) = editor.selected.and_then(|i| scene.position(i)) {
            out.extend(build_rotate_arcs(center, editor));
        }
    }
    if editor.selected == Some(scene::SCENE_CUBE) {
        if let Some(position) = scene.position(scene::SCENE_CUBE) {
            let rotation = scene
                .rotation(scene::SCENE_CUBE)
                .unwrap_or(glam::Quat::IDENTITY);
            out.extend(build_cube_edges(
                position,
                rotation,
                glam::Vec3::new(1.0, 0.6, 0.1),
            ));
        }
    }
    out
}

/// Static rotate dials: 3 colored 300° arcs (Blender `ED_GIZMO_DIAL_DRAW_FLAG_CLIP`
/// style with staggered gaps) around the selected object. The white
/// trackball ring and the sweep indicator are view-dependent and are
/// rebuilt every frame in `render` (small fixed buffer, no allocation).
fn build_rotate_arcs(center: glam::Vec3, editor: &Editor) -> Vec<Vertex> {
    let mut out = Vec::with_capacity(150);
    let colors = [
        glam::Vec3::new(1.0, 0.25, 0.25),
        glam::Vec3::new(0.3, 1.0, 0.3),
        glam::Vec3::new(0.3, 0.55, 1.0),
    ];
    let active = editor
        .rotate_dragging
        .and_then(|drag| match drag.kind {
            editor::RotateKind::Axis(axis) => Some(axis.index()),
            editor::RotateKind::Trackball => None,
        });
    for (index, color) in colors.iter().enumerate() {
        let kind = editor::RotateKind::Axis(match index {
            0 => editor::Axis::X,
            1 => editor::Axis::Y,
            _ => editor::Axis::Z,
        });
        let tint = if active == Some(index) {
            glam::Vec3::splat(1.0)
        } else {
            *color * 0.85
        };
        let points = editor::ring_points(center, kind, 48);
        for window in points.windows(2) {
            push_line(&mut out, window[0], window[1], tint);
        }
    }
    out
}

type EngineHandle = u64;

#[derive(Debug)]
struct NativeWindow(NonNull<ndk_sys::ANativeWindow>);

impl NativeWindow {
    /// SAFETY: the Java surface must be alive for this call and the caller
    /// transfers the acquired-reference ownership to this wrapper.
    unsafe fn from_surface(env: JNIEnv<'_>, surface: JObject<'_>) -> Option<Self> {
        if surface.is_null() {
            return None;
        }
        let raw = ndk_sys::ANativeWindow_fromSurface(env.get_raw(), surface.as_raw())
            as *mut ndk_sys::ANativeWindow;
        let ptr = NonNull::new(raw)?;
        Some(Self(ptr))
    }

    fn size(&self) -> (u32, u32) {
        let width = unsafe { ndk_sys::ANativeWindow_getWidth(self.0.as_ptr()) };
        let height = unsafe { ndk_sys::ANativeWindow_getHeight(self.0.as_ptr()) };
        (width.max(0) as u32, height.max(0) as u32)
    }

    fn raw_handle(&self) -> AndroidNdkWindowHandle {
        AndroidNdkWindowHandle::new(self.0.cast::<c_void>())
    }
}

impl Clone for NativeWindow {
    fn clone(&self) -> Self {
        unsafe {
            ndk_sys::ANativeWindow_acquire(self.0.as_ptr());
        }
        Self(self.0)
    }
}

impl Drop for NativeWindow {
    fn drop(&mut self) {
        unsafe {
            ndk_sys::ANativeWindow_release(self.0.as_ptr());
        }
    }
}

unsafe impl Send for NativeWindow {}
unsafe impl Sync for NativeWindow {}

impl HasWindowHandle for NativeWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        unsafe {
            Ok(WindowHandle::borrow_raw(
                RawWindowHandle::AndroidNdk(self.raw_handle()),
            ))
        }
    }
}

impl HasDisplayHandle for NativeWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        unsafe {
            Ok(DisplayHandle::borrow_raw(RawDisplayHandle::Android(
                AndroidDisplayHandle::new(),
            )))
        }
    }
}

#[derive(Clone)]
struct EngineStatus {
    state: &'static str,
    detail: String,
    frames: u64,
    fps: f32,
    size: (u32, u32),
    adapter: String,
    backend: String,
    format: String,
    present: String,
    camera: String,
}

impl EngineStatus {
    fn initializing(id: u64) -> Self {
        Self {
            state: "initializing",
            detail: format!("engine {id}"),
            frames: 0,
            fps: 0.0,
            size: (0, 0),
            adapter: String::new(),
            backend: String::new(),
            format: String::new(),
            present: String::new(),
            camera: String::new(),
        }
    }
}

struct WorkerState {
    status: EngineStatus,
    frame_pending: bool,
    resize_pending: Option<(u32, u32)>,
    shutdown: bool,
    camera: CameraRig,
    camera_commands: VecDeque<CameraCommand>,
    scene: Scene,
    editor: Editor,
    scene_dirty: bool,
}

struct EngineSlot {
    id: EngineHandle,
    shared: Arc<Shared>,
}

struct Shared {
    state: Mutex<WorkerState>,
    ready: Condvar,
}

impl Shared {
    fn new(id: EngineHandle, size: (u32, u32)) -> Self {
        let mut status = EngineStatus::initializing(id);
        status.size = size;
        Self {
            state: Mutex::new(WorkerState {
                status,
                frame_pending: false,
                resize_pending: None,
                shutdown: false,
                camera: CameraRig::default(),
                camera_commands: VecDeque::new(),
                scene: Scene::default_scene(DEFAULT_CAMERA_POSITION, DEFAULT_LIGHT_POSITION),
                editor: Editor::new(),
                scene_dirty: true,
            }),
            ready: Condvar::new(),
        }
    }

    fn request_frame(&self) -> bool {
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown {
            return false;
        }
        let already_pending = state.frame_pending;
        state.frame_pending = true;
        self.ready.notify_one();
        !already_pending
    }

    fn request_resize(&self, width: u32, height: u32) -> bool {
        if width == 0 || height == 0 {
            return false;
        }
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown {
            return false;
        }
        state.resize_pending = Some((width, height));
        state.frame_pending = true;
        self.ready.notify_one();
        true
    }

    fn request_camera_command(&self, command: CameraCommand) -> bool {
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown {
            return false;
        }
        state.camera_commands.push_back(command);
        state.frame_pending = true;
        self.ready.notify_one();
        true
    }

    fn request_orbit(&self, dx: f32, dy: f32) -> bool {
        self.request_camera_command(CameraCommand::Orbit { dx, dy })
    }

    fn request_pan(&self, dx: f32, dy: f32) -> bool {
        self.request_camera_command(CameraCommand::Pan { dx, dy })
    }

    fn request_zoom(&self, scale: f32) -> bool {
        if !(scale > 0.0) {
            return false;
        }
        self.request_camera_command(CameraCommand::Zoom { scale })
    }

    fn request_frame_all(&self) -> bool {
        self.request_camera_command(CameraCommand::FrameAll)
    }

    fn wake(&self, state: &mut WorkerState) {
        state.frame_pending = true;
        self.ready.notify_one();
    }

    fn set_tool(&self, tool: Tool) -> bool {
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown {
            return false;
        }
        if state.editor.tool == tool {
            return true;
        }
        state.editor.tool = tool;
        self.wake(&mut state);
        true
    }

    /// Tap pick. Returns the selected object index, or -1 on a miss.
    fn pick(&self, px: f32, py: f32) -> i32 {
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown {
            return -1;
        }
        let size = state.status.size;
        let camera = state.camera;
        let (eye, dir) = Editor::ray_from_screen(&camera, size, px, py);
        let WorkerState { editor, scene, .. } = &mut *state;
        editor.apply_tap_pick(scene, eye, dir);
        let selected = state.editor.selected.map(|i| i as i32).unwrap_or(-1);
        state.scene_dirty = true;
        self.wake(&mut state);
        selected
    }

    fn gizmo_hit(&self, px: f32, py: f32) -> i32 {
        let state = self.state.lock().expect("engine state mutex");
        if state.shutdown || !state.editor.show_gizmo() {
            return -1;
        }
        let size = state.status.size;
        state
            .editor
            .axis_hit(&state.scene, &state.camera, size, px, py)
            .map(|axis| axis.index() as i32)
            .unwrap_or(-1)
    }

    fn move_begin(&self, px: f32, py: f32) -> i32 {
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown || state.editor.tool != Tool::Move || state.editor.selected.is_none() {
            return -1;
        }
        let size = state.status.size;
        let camera = state.camera;
        let WorkerState { editor, scene, .. } = &mut *state;
        let axis = editor.begin_drag(scene, &camera, size, px, py);
        self.wake(&mut state);
        axis.map(|a| a.index() as i32).unwrap_or(-1)
    }

    fn move_to(&self, px: f32, py: f32) -> bool {
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown || state.editor.dragging.is_none() {
            return false;
        }
        let size = state.status.size;
        let camera = state.camera;
        let WorkerState { editor, scene, .. } = &mut *state;
        let changed = editor.drag_to(scene, &camera, size, px, py);
        if changed {
            state.scene_dirty = true;
            self.wake(&mut state);
        }
        true
    }

    fn move_end(&self) -> bool {
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown {
            return false;
        }
        state.editor.end_drag();
        self.wake(&mut state);
        true
    }

    fn rotate_begin(&self, px: f32, py: f32) -> i32 {
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown || state.editor.tool != Tool::Rotate || state.editor.selected.is_none() {
            return -1;
        }
        let size = state.status.size;
        let camera = state.camera;
        let WorkerState { editor, scene, .. } = &mut *state;
        let kind = editor.rotate_begin(scene, &camera, size, px, py);
        self.wake(&mut state);
        kind.map(|k| k.index()).unwrap_or(-1)
    }

    fn rotate_to(&self, px: f32, py: f32) -> bool {
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown || state.editor.rotate_dragging.is_none() {
            return false;
        }
        let size = state.status.size;
        let camera = state.camera;
        let WorkerState { editor, scene, .. } = &mut *state;
        let changed = editor.rotate_to(scene, &camera, size, px, py);
        if changed {
            state.scene_dirty = true;
            self.wake(&mut state);
        }
        true
    }

    fn rotate_end(&self) -> bool {
        let mut state = self.state.lock().expect("engine state mutex");
        if state.shutdown {
            return false;
        }
        state.editor.rotate_end();
        self.wake(&mut state);
        true
    }

    fn request_shutdown(&self) {
        let mut state = self.state.lock().expect("engine state mutex");
        state.shutdown = true;
        state.frame_pending = false;
        state.resize_pending = None;
        state.camera_commands.clear();
        self.ready.notify_all();
    }

    fn snapshot(&self) -> EngineStatus {
        self.state.lock().expect("engine state mutex").status.clone()
    }
}

static ENGINES: OnceLock<Mutex<HashMap<EngineHandle, Arc<EngineSlot>>>> =
    OnceLock::new();
static NEXT_ENGINE: AtomicU64 = AtomicU64::new(1);

fn engines() -> &'static Mutex<HashMap<EngineHandle, Arc<EngineSlot>>> {
    ENGINES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn find_engine(handle: EngineHandle) -> Option<Arc<EngineSlot>> {
    if handle == 0 {
        return None;
    }
    engines()
        .lock()
        .expect("engine registry mutex")
        .get(&handle)
        .cloned()
}

struct SurfaceContext {
    surface: wgpu::Surface<'static>,
    owner: NativeWindow,
    config: wgpu::SurfaceConfiguration,
}

struct Renderer {
    context: SurfaceContext,
    device: wgpu::Device,
    queue: wgpu::Queue,
    triangle_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    light_pipeline: wgpu::RenderPipeline,
    gizmo_pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    triangle_buffer: wgpu::Buffer,
    line_buffer: wgpu::Buffer,
    light_buffer: wgpu::Buffer,
    gizmo_buffer: wgpu::Buffer,
    view_buffer: wgpu::Buffer,
    line_count: u32,
    light_count: u32,
    gizmo_count: u32,
    light_center: glam::Vec3,
    depth: wgpu::TextureView,
    adapter_name: String,
    backend: String,
    format: String,
    present: String,
}

impl Renderer {
    async fn new(
        owner: NativeWindow,
        size: (u32, u32),
        scene: &Scene,
    ) -> Result<Self, String> {
        let mut instance_descriptor =
            wgpu::InstanceDescriptor::new_with_display_handle(Box::new(owner.clone()));
        instance_descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(instance_descriptor);
        let target = wgpu::SurfaceTarget::Window(Box::new(owner.clone()));
        let surface = instance
            .create_surface(target)
            .map_err(|error| format!("wgpu surface creation failed: {error}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|error| format!("no Vulkan adapter: {error}"))?;

        let info = adapter.get_info();
        log::info!(
            "K3Test phase1 adapter={} backend={:?} driver={:?}",
            info.name, info.backend, info.driver
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|error| format!("Vulkan device creation failed: {error}"))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|format| *format == wgpu::TextureFormat::Rgba8UnormSrgb)
            .or_else(|| caps.formats.iter().copied().find(|format| format.is_srgb()))
            .unwrap_or(caps.formats[0]);
        let present = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            caps.present_modes[0]
        };

        let size = if size.0 > 0 && size.1 > 0 {
            size
        } else {
            owner.size()
        };
        let size = (size.0.max(1), size.1.max(1));
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.0,
            height: size.1,
            present_mode: present,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![format],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);
        let depth = create_depth_view(&device, size);

        // Capture validation errors instead of letting wgpu's fatal
        // handler kill the render thread silently.
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("K3Test Phase 1 viewport shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{VERTEX_SHADER}\n{FRAGMENT_SHADER}").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K3Test Phase 1 BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        // VP at 0, light center/right/up at 256-byte steps: keeps every
        // uniform binding offset aligned for any device (PowerVR 32,
        // Mali/Adreno up to 256).
        let mut uniform_bytes = [0f32; 196];
        uniform_bytes[0] = 1.0;
        uniform_bytes[5] = 1.0;
        uniform_bytes[10] = 1.0;
        uniform_bytes[15] = 1.0;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K3Test Phase 1 uniform"),
            contents: bytemuck::cast_slice(&uniform_bytes),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_binding = |offset| {
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform,
                    offset,
                    size: None,
                }),
            }
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("K3Test Phase 1 BG"),
            layout: &bind_group_layout,
            entries: &[
                uniform_binding(0),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 256,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 512,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 768,
                        size: None,
                    }),
                },
            ],
        });
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("K3Test Phase 1 pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                ..Default::default()
            });

        // Fixed-capacity dynamic buffers: rewritten only when the scene
        // or selection changes, never per frame while idle.
        const LINE_MAX: usize = 512;
        const LIGHT_MAX: usize = 64;
        let zero_vertex = Vertex {
            position: [0.0; 3],
            color: [0.0; 3],
        };
        let cube_offset = scene
            .position(scene::SCENE_CUBE)
            .unwrap_or(glam::Vec3::ZERO);
        let cube_rotation = scene
            .rotation(scene::SCENE_CUBE)
            .unwrap_or(glam::Quat::IDENTITY);
        let cube = build_cube_vertices(cube_offset, cube_rotation);
        let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K3Test cube (2 m)"),
            contents: bytemuck::cast_slice(&cube),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K3Test grid/camera/gizmo/selection lines"),
            contents: bytemuck::cast_slice(&vec![zero_vertex; LINE_MAX]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K3Test billboard light icon (local XY)"),
            contents: bytemuck::cast_slice(&vec![zero_vertex; LIGHT_MAX]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let gizmo_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K3Test move gizmo"),
            contents: bytemuck::cast_slice(&vec![zero_vertex; 256]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let view_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K3Test view-dependent lines (trackball, sweep)"),
            contents: bytemuck::cast_slice(&vec![zero_vertex; 160]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 3 * 4,
                    shader_location: 1,
                },
            ],
        };

        fn make_pipeline(
            device: &wgpu::Device,
            shader: &wgpu::ShaderModule,
            layout: &wgpu::PipelineLayout,
            vertex_layout: &wgpu::VertexBufferLayout<'_>,
            format: wgpu::TextureFormat,
            topology: wgpu::PrimitiveTopology,
            cull_mode: Option<wgpu::Face>,
            entry_point: &'static str,
            depth_compare: wgpu::CompareFunction,
            depth_write: bool,
            label: &'static str,
        ) -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(vertex_layout.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    cull_mode,
                    front_face: wgpu::FrontFace::Ccw,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(depth_compare),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        }

        let triangle_pipeline = make_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            &vertex_layout,
            format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "vs_main",
            wgpu::CompareFunction::Less,
            true,
            "K3Test cube triangles",
        );
        let line_pipeline = make_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            &vertex_layout,
            format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "vs_main",
            wgpu::CompareFunction::Less,
            true,
            "K3Test grid and camera",
        );
        let light_pipeline = make_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            &vertex_layout,
            format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "vs_light",
            wgpu::CompareFunction::Less,
            true,
            "K3Test billboard light icon",
        );
        // Gizmo arrows render as an overlay (Blender-style): visible
        // through the object so manipulators are never buried in it.
        // Solid triangles: GPU lines are always 1 px wide.
        let gizmo_pipeline = make_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            &vertex_layout,
            format,
            wgpu::PrimitiveTopology::TriangleList,
            None,
            "vs_main",
            wgpu::CompareFunction::Always,
            false,
            "K3Test move gizmo overlay",
        );
        if let Some(error) = error_scope.pop().await {
            return Err(format!("wgpu validation: {error:?}"));
        }

        let mut renderer = Self {
            context: SurfaceContext {
                surface,
                owner,
                config,
            },
            device,
            queue,
            triangle_pipeline,
            line_pipeline,
            light_pipeline,
            gizmo_pipeline,
            uniform,
            bind_group,
            triangle_buffer,
            line_buffer,
            light_buffer,
            gizmo_buffer,
            view_buffer,
            line_count: 0,
            light_count: 0,
            gizmo_count: 0,
            light_center: scene
                .position(scene::SCENE_LIGHT)
                .unwrap_or(DEFAULT_LIGHT_POSITION),
            depth,
            adapter_name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            format: format!("{format:?}"),
            present: format!("{present:?}"),
        };
        renderer.sync_scene(scene, &Editor::new());
        Ok(renderer)
    }

    /// Re-upload object/selection/gizmo geometry after a scene or
    /// editor state change. No allocation escapes this call beyond the
    /// small temporary vectors.
    fn sync_scene(&mut self, scene: &Scene, editor: &Editor) {
        let cube_offset = scene
            .position(scene::SCENE_CUBE)
            .unwrap_or(glam::Vec3::ZERO);
        let cube_rotation = scene
            .rotation(scene::SCENE_CUBE)
            .unwrap_or(glam::Quat::IDENTITY);
        let cube = build_cube_vertices(cube_offset, cube_rotation);
        self.queue
            .write_buffer(&self.triangle_buffer, 0, bytemuck::cast_slice(&cube));

        let lines = build_world_lines(scene, editor);
        self.queue
            .write_buffer(&self.line_buffer, 0, bytemuck::cast_slice(&lines));
        self.line_count = lines.len() as u32;

        let light_color = if editor.selected == Some(scene::SCENE_LIGHT) {
            glam::Vec3::new(1.0, 0.6, 0.1)
        } else {
            glam::Vec3::new(1.0, 0.85, 0.25)
        };
        let light = build_light_glyph_vertices(light_color);
        self.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&light));
        self.light_count = light.len() as u32;

        let gizmo: Vec<Vertex> = if editor.show_gizmo() {
            editor
                .selected
                .and_then(|index| scene.position(index))
                .map(editor::build_gizmo_vertices)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        self.queue
            .write_buffer(&self.gizmo_buffer, 0, bytemuck::cast_slice(&gizmo));
        self.gizmo_count = gizmo.len() as u32;

        self.light_center = scene
            .position(scene::SCENE_LIGHT)
            .unwrap_or(DEFAULT_LIGHT_POSITION);
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.context.config.width = width;
        self.context.config.height = height;
        self.context
            .surface
            .configure(&self.device, &self.context.config);
        self.depth = create_depth_view(&self.device, (width, height));
    }

    fn update_uniform(&self, camera: CameraRig, aspect: f32) {
        let view = camera.view_matrix();
        let projection =
            glam::Mat4::perspective_rh(VIEW_FOV_Y, aspect, VIEW_NEAR, VIEW_FAR);
        let view_proj = projection * view;
        let columns = view_proj.to_cols_array_2d();
        let mut flat = [0_f32; 196];
        for (column_index, column) in columns.iter().enumerate() {
            for row in 0..4 {
                flat[column_index * 4 + row] = column[row];
            }
        }
        // Screen-aligned light icon axes (same convention as look_at_rh).
        let eye = camera.eye();
        let forward = (camera.target - eye).normalize();
        let right = forward.cross(WORLD_UP).normalize();
        let up = right.cross(forward).normalize();
        flat[64..67].copy_from_slice(&self.light_center.to_array());
        flat[128..131].copy_from_slice(&right.to_array());
        flat[192..195].copy_from_slice(&up.to_array());
        self.queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&flat));
    }

    /// View-dependent lines: Blender's white trackball ring (view-aligned
    /// dial, `MAN_AXIS_ROT_T`) plus the drag sweep indicator (white radius
    /// line + dashed start radius, `WM_GIZMO_DRAW_VALUE`).
    /// Fixed-size stack buffer: no allocation per frame.
    fn sync_view_lines(
        &self,
        camera: CameraRig,
        editor: &Editor,
        scene: &Scene,
    ) -> u32 {
        let mut lines = [Vertex {
            position: [0.0; 3],
            color: [0.0; 3],
        }; 160];
        let mut count: usize = 0;
        if editor.tool != Tool::Rotate {
            return 0;
        }
        let Some(center) = editor.selected.and_then(|i| scene.position(i)) else {
            return 0;
        };
        let eye = camera.eye();
        let forward = (camera.target - eye).normalize();
        let right = forward.cross(WORLD_UP).normalize();
        let up = right.cross(forward).normalize();
        let radius = editor::ROTATE_RING_RADIUS;
        // White trackball ring.
        let steps = 48;
        for i in 0..steps {
            let a0 =
                i as f32 * std::f32::consts::TAU / steps as f32;
            let a1 =
                (i + 1) as f32 * std::f32::consts::TAU / steps as f32;
            let p0 = center + (right * a0.cos() + up * a0.sin()) * radius;
            let p1 = center + (right * a1.cos() + up * a1.sin()) * radius;
            let white = glam::Vec3::splat(0.92);
            lines[count] = Vertex::new(p0, white);
            lines[count + 1] = Vertex::new(p1, white);
            count += 2;
        }
        // Sweep indicator while rotating by axis.
        if let Some(drag) = editor.rotate_dragging {
            let grab = center + drag.grab_vec * radius;
            let current = center + drag.current_vec * radius;
            // Solid: center -> current angle.
            lines[count] = Vertex::new(center, glam::Vec3::splat(1.0));
            lines[count + 1] = Vertex::new(current, glam::Vec3::splat(1.0));
            count += 2;
            // Dashed: center -> grab angle (3 dashes).
            for (t0, t1) in [(0.0, 0.4), (0.5, 0.7), (0.8, 1.0)] {
                let dash_color = glam::Vec3::splat(0.85);
                lines[count] = Vertex::new(
                    center + (grab - center) * t0,
                    dash_color,
                );
                lines[count + 1] = Vertex::new(
                    center + (grab - center) * t1,
                    dash_color,
                );
                count += 2;
            }
        }
        self.queue
            .write_buffer(&self.view_buffer, 0, bytemuck::cast_slice(&lines[..count]));
        count as u32
    }

    fn render(
        &mut self,
        camera: CameraRig,
        editor: &Editor,
        scene: &Scene,
    ) -> Result<(), String> {
        let started = Instant::now();
        let frame = match self.context.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.context
                    .surface
                    .configure(&self.device, &self.context.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err("surface lost".to_string());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("swapchain validation error".to_string());
            }
        };

        let width = self.context.config.width.max(1) as f32;
        let height = self.context.config.height.max(1) as f32;
        self.update_uniform(camera, width / height);
        let view_line_count = self.sync_view_lines(camera, editor, scene);

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("K3Test Phase 1 frame encoder"),
            },
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("K3Test Phase 1 frame pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.10,
                            g: 0.11,
                            b: 0.14,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                ..Default::default()
            });
            pass.set_bind_group(0, &self.bind_group, &[]);

            // View overlay first: grid + camera, then the billboarded
            // light icon — all in this same tiled render pass.
            pass.set_pipeline(&self.line_pipeline);
            pass.set_vertex_buffer(0, self.line_buffer.slice(..));
            pass.draw(0..self.line_count, 0..1);
            pass.set_pipeline(&self.light_pipeline);
            pass.set_vertex_buffer(0, self.light_buffer.slice(..));
            pass.draw(0..self.light_count, 0..1);
            if view_line_count > 0 {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_vertex_buffer(0, self.view_buffer.slice(..));
                pass.draw(0..view_line_count, 0..1);
            }

            // Object triangles, then the gizmo overlay on top.
            pass.set_pipeline(&self.triangle_pipeline);
            pass.set_vertex_buffer(0, self.triangle_buffer.slice(..));
            pass.draw(0..36, 0..1);
            if self.gizmo_count > 0 {
                pass.set_pipeline(&self.gizmo_pipeline);
                pass.set_vertex_buffer(0, self.gizmo_buffer.slice(..));
                pass.draw(0..self.gizmo_count, 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
        let _ = started;
        Ok(())
    }
}

fn create_depth_view(device: &wgpu::Device, size: (u32, u32)) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("K3Test depth"),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn render_worker(shared: Arc<Shared>, owner: NativeWindow) {
    let mut renderer: Option<Renderer> = None;
    let mut fps_frames: u64 = 0;
    let mut last_fps = Instant::now();
    let mut last_status = Instant::now();

    let initial_size = owner.size();
    {
        let state = shared.state.lock().expect("engine state mutex");
        if state.shutdown {
            drop(owner);
            return;
        }
    }

    let initial_scene = {
        shared
            .state
            .lock()
            .expect("engine state mutex")
            .scene
            .clone()
    };
    match pollster::block_on(Renderer::new(owner, initial_size, &initial_scene)) {
        Ok(candidate) => {
            let mut state = shared.state.lock().expect("engine state mutex");
            if state.shutdown {
                drop(candidate);
                return;
            }
            state.status.state = "running";
            state.status.adapter = candidate.adapter_name.clone();
            state.status.backend = candidate.backend.clone();
            state.status.format = candidate.format.clone();
            state.status.present = candidate.present.clone();
            state.status.size = (
                candidate.context.config.width,
                candidate.context.config.height,
            );
            renderer = Some(candidate);
        }
        Err(error) => {
            let mut state = shared.state.lock().expect("engine state mutex");
            state.status.state = "error";
            state.status.detail = error.clone();
            log::error!("K3Test Phase 1 initialization failed: {error}");
            return;
        }
    }

    loop {
        let action = {
            let mut state = shared.state.lock().expect("engine state mutex");
            if state.shutdown {
                break;
            }
            while !state.frame_pending
                && state.resize_pending.is_none()
                && state.camera_commands.is_empty()
                && !state.shutdown
            {
                state = shared
                    .ready
                    .wait(state)
                    .expect("engine state mutex");
            }
            if state.shutdown {
                break;
            }
            let resize = state.resize_pending.take();
            let frame_pending = state.frame_pending;
            state.frame_pending = false;
            let scene_dirty = state.scene_dirty;
            state.scene_dirty = false;
            let commands: Vec<CameraCommand> = state.camera_commands.drain(..).collect();
            (
                resize,
                frame_pending,
                commands,
                state.camera,
                scene_dirty,
                state.scene.clone(),
                state.editor.clone(),
            )
        };

        let (resize, frame_pending, commands, mut camera, scene_dirty, scene_snapshot, editor_snapshot) =
            action;
        if let Some((width, height)) = resize {
            if let Some(current) = renderer.as_mut() {
                current.resize(width, height);
                let mut state = shared.state.lock().expect("engine state mutex");
                state.status.size = (width, height);
            }
        }

        for command in commands {
            let size = renderer
                .as_ref()
                .map(|current| {
                    (
                        current.context.config.width,
                        current.context.config.height,
                    )
                })
                .unwrap_or((1, 1));
            camera.apply(command, size);
        }
        {
            let mut state = shared.state.lock().expect("engine state mutex");
            state.camera = camera;
        }

        if !frame_pending {
            continue;
        }
        let Some(current) = renderer.as_mut() else {
            continue;
        };
        if scene_dirty {
            current.sync_scene(&scene_snapshot, &editor_snapshot);
        }
        match current.render(camera, &editor_snapshot, &scene_snapshot) {
            Ok(()) => {
                fps_frames += 1;
                let mut state = shared.state.lock().expect("engine state mutex");
                state.status.frames = state.status.frames.saturating_add(1);
                state.status.detail = "frames".to_string();
                if last_fps.elapsed() >= Duration::from_secs(1) {
                    state.status.fps =
                        fps_frames as f32 / last_fps.elapsed().as_secs_f32();
                    fps_frames = 0;
                    last_fps = Instant::now();
                }
                if last_status.elapsed() >= Duration::from_secs(2) {
                    last_status = Instant::now();
                    log::info!(
                        "K3Test Phase 1 running backend={} fps={:.1}",
                        current.backend,
                        state.status.fps
                    );
                }
            }
            Err(error) => {
                let mut state = shared.state.lock().expect("engine state mutex");
                state.status.state = "error";
                state.status.detail = error;
            }
        }
    }

    // Drop GPU resources, then wgpu surface, then ANativeWindow on this thread.
    drop(renderer);
}

fn init_logging() {
    let logger = android_logger::Config::default()
        .with_max_level(log::LevelFilter::Info)
        .with_tag("K3TestPhase1");
    android_logger::init_once(logger);
    log::info!("K3Test Phase 1 logging ready");
}

fn jni_string(env: &mut JNIEnv<'_>, value: &str) -> jstring {
    env.new_string(value)
        .map(jni::objects::JString::into_raw)
        .unwrap_or_else(|_| std::ptr::null_mut())
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeCreate(
    mut env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    surface: JObject<'_>,
) -> jlong {
    init_logging();
    if surface.is_null() {
        log::error!("K3Test Phase 1 nativeCreate called with a null Surface");
        return 0;
    }
    let Some(window) = (unsafe { NativeWindow::from_surface(env, surface) }) else {
        log::error!("K3Test Phase 1 could not acquire ANativeWindow");
        return 0;
    };

    let id = NEXT_ENGINE.fetch_add(1, Ordering::Relaxed);
    let shared = Arc::new(Shared::new(id, window.size()));
    let slot = Arc::new(EngineSlot {
        id,
        shared: Arc::clone(&shared),
    });
    engines()
        .lock()
        .expect("engine registry mutex")
        .insert(id, Arc::clone(&slot));

    let thread_shared = Arc::clone(&shared);
    match thread::Builder::new()
        .name(format!("K3Bridge-{id}"))
        .spawn(move || {
            render_worker(thread_shared, window);
        }) {
        Ok(_) => {
            log::info!("K3Test Phase 1 engine {id} worker started");
            id as jlong
        }
        Err(error) => {
            engines()
                .lock()
                .expect("engine registry mutex")
                .remove(&id);
            log::error!("K3Test Phase 1 render thread spawn failed: {error}");
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeResize(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    width: i32,
    height: i32,
) -> jni::sys::jboolean {
    if handle <= 0 || width <= 0 || height <= 0 {
        return 0;
    }
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    if engine.shared.request_resize(width as u32, height as u32) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeFrame(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
) -> jni::sys::jboolean {
    if handle <= 0 {
        return 0;
    }
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    if engine.shared.request_frame() {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeOrbit(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    dx: jni::sys::jfloat,
    dy: jni::sys::jfloat,
) -> jni::sys::jboolean {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    if engine.shared.request_orbit(dx, dy) { 1 } else { 0 }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativePan(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    dx: jni::sys::jfloat,
    dy: jni::sys::jfloat,
) -> jni::sys::jboolean {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    if engine.shared.request_pan(dx, dy) { 1 } else { 0 }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeZoom(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    scale: jni::sys::jfloat,
) -> jni::sys::jboolean {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    if engine.shared.request_zoom(scale) { 1 } else { 0 }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeFrameAll(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
) -> jni::sys::jboolean {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    if engine.shared.request_frame_all() { 1 } else { 0 }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeSetTool(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    tool: jni::sys::jint,
) -> jni::sys::jboolean {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    let selected = engine.shared.set_tool(if tool == 1 {
        Tool::Move
    } else {
        Tool::None
    });
    if selected { 1 } else { 0 }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativePick(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    x: jni::sys::jfloat,
    y: jni::sys::jfloat,
) -> jni::sys::jint {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return -1;
    };
    engine.shared.pick(x, y)
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeGizmoHit(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    x: jni::sys::jfloat,
    y: jni::sys::jfloat,
) -> jni::sys::jint {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return -1;
    };
    engine.shared.gizmo_hit(x, y)
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeMoveBegin(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    x: jni::sys::jfloat,
    y: jni::sys::jfloat,
) -> jni::sys::jint {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return -1;
    };
    engine.shared.move_begin(x, y)
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeMoveTo(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    x: jni::sys::jfloat,
    y: jni::sys::jfloat,
) -> jni::sys::jboolean {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    if engine.shared.move_to(x, y) { 1 } else { 0 }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeMoveEnd(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
) -> jni::sys::jboolean {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    if engine.shared.move_end() { 1 } else { 0 }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeRotateBegin(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    x: jni::sys::jfloat,
    y: jni::sys::jfloat,
) -> jni::sys::jint {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return -1;
    };
    engine.shared.rotate_begin(x, y)
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeRotateTo(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
    x: jni::sys::jfloat,
    y: jni::sys::jfloat,
) -> jni::sys::jboolean {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    if engine.shared.rotate_to(x, y) { 1 } else { 0 }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeRotateEnd(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
) -> jni::sys::jboolean {
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return 0;
    };
    if engine.shared.rotate_end() { 1 } else { 0 }
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeDestroy(
    _env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
) -> jni::sys::jboolean {
    if handle <= 0 {
        return 1;
    }
    let id = handle as EngineHandle;
    let removed = engines()
        .lock()
        .expect("engine registry mutex")
        .remove(&id);
    let Some(engine) = removed else {
        // Double destroy is not an error: it must never free twice.
        return 1;
    };
    // Non-blocking teardown. The render worker performs the ordered cleanup
    // on its own thread; the UI thread is never joined here.
    engine.shared.request_shutdown();
    log::info!("K3Test Phase 1 engine {id} shutdown requested");
    1
}

#[no_mangle]
pub extern "system" fn Java_dev_k3test_NativeBridge_nativeInfo(
    mut env: JNIEnv<'_>,
    _receiver: JObject<'_>,
    handle: jlong,
) -> jstring {
    if handle <= 0 {
        return jni_string(&mut env, "state=none");
    }
    let Some(engine) = find_engine(handle as EngineHandle) else {
        return jni_string(&mut env, "state=none");
    };
    let status = engine.shared.snapshot();
    let state = engine.shared.state.lock().expect("engine state mutex");
    let tool = state.editor.tool.name();
    let selected = state.editor.selected.map(|i| i as i32).unwrap_or(-1);
    let cube = state
        .scene
        .position(scene::SCENE_CUBE)
        .unwrap_or(glam::Vec3::ZERO);
    let detail = if state.editor.rotate_dragging.is_some() {
        format!("rotate={:.1}°", state.editor.rotate_degrees())
    } else {
        status.detail.clone()
    };
    jni_string(
        &mut env,
        &format!(
            "state={} frames={} fps={:.1} adapter={} backend={} format={} present={} size={}x{} tool={} selected={} cube=({:.2},{:.2},{:.2}) detail={}",
            status.state,
            status.frames,
            status.fps,
            status.adapter,
            status.backend,
            status.format,
            status.present,
            status.size.0,
            status.size.1,
            tool,
            selected,
            cube.x,
            cube.y,
            cube.z,
            detail
        ),
    )
}
