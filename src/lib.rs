//! K3Test spike — Compose/JNI bridge + wgpu 30.
//!
//! The important part is not the cube. It is the ownership contract:
//! * Kotlin creates the `SurfaceView` and owns its visible lifetime.
//! * Rust acquires an `ANativeWindow` with `ANativeWindow_fromSurface`,
//!   which increments the native reference.
//! * `wgpu::Surface` is created and destroyed only on the native render
//!   thread. The Java/Kotlin callbacks never wait for GPU teardown.
//! * The `wgpu::Surface` is dropped before the retained `ANativeWindow`
//!   is released, and every acquire is paired with exactly one release.

use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{
        Arc, Condvar, Mutex, OnceLock,
    },
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
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

const VERTEX_SHADER: &str = r"
@group(0) @binding(0) var<uniform> model: mat4x4<f32>;
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
    out.clip_position = model * vec4<f32>(input.position, 1.0);
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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

const CUBE_VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5, -0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [0.5, -0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [0.5, 0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [-0.5, -0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [0.5, 0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [-0.5, 0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [-0.5, 0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [0.5, 0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [0.5, 0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [0.5, -0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [0.5, -0.5, -0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, -0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, 0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [0.5, -0.5, -0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, 0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [0.5, -0.5, 0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, -0.5, 0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, 0.5, 0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, 0.5, 0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, 0.5, -0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, 0.5, -0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [-0.5, 0.5, 0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, 0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [-0.5, 0.5, -0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, 0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, -0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.2, 1.0, 1.0] },
    Vertex { position: [0.5, -0.5, -0.5], color: [0.2, 1.0, 1.0] },
    Vertex { position: [0.5, -0.5, 0.5], color: [0.2, 1.0, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.2, 1.0, 1.0] },
    Vertex { position: [0.5, -0.5, 0.5], color: [0.2, 1.0, 1.0] },
    Vertex { position: [-0.5, -0.5, 0.5], color: [0.2, 1.0, 1.0] },
];

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
    frame_ms: f32,
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
            frame_ms: 0.0,
        }
    }
}

struct WorkerState {
    status: EngineStatus,
    frame_pending: bool,
    resize_pending: Option<(u32, u32)>,
    shutdown: bool,
}

struct EngineSlot {
    id: EngineHandle,
    shared: Arc<Shared>,
}

struct Shared {
    state: Mutex<WorkerState>,
    ready: Condvar,
    frame_requested: AtomicBool,
}

impl Shared {
    fn new(id: EngineHandle) -> Self {
        Self {
            state: Mutex::new(WorkerState {
                status: EngineStatus::initializing(id),
                frame_pending: false,
                resize_pending: None,
                shutdown: false,
            }),
            ready: Condvar::new(),
            frame_requested: AtomicBool::new(false),
        }
    }

    fn request_frame(&self) -> bool {
        if self.frame_requested.swap(true, Ordering::AcqRel) {
            return false;
        }
        let mut state = self.state.lock().expect("engine state mutex");
        state.frame_pending = true;
        self.ready.notify_one();
        true
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
        self.ready.notify_one();
        true
    }

    fn request_shutdown(&self) {
        let mut state = self.state.lock().expect("engine state mutex");
        state.shutdown = true;
        state.frame_pending = false;
        state.resize_pending = None;
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
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    depth: wgpu::TextureView,
    adapter_name: String,
    backend: String,
    format: String,
    present: String,
}

impl Renderer {
    async fn new(owner: NativeWindow, size: (u32, u32)) -> Result<Self, String> {
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
            "K3Test bridge adapter={} backend={:?} driver={:?}",
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("K3Test bridge shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{VERTEX_SHADER}\n{FRAGMENT_SHADER}").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K3Test bridge BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K3Test bridge uniform"),
            contents: bytemuck::cast_slice(&[0f32; 64]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("K3Test bridge BG"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("K3Test bridge pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                ..Default::default()
            });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K3Test bridge cube"),
            contents: bytemuck::cast_slice(CUBE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("K3Test bridge pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
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
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
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
        });

        Ok(Self {
            context: SurfaceContext {
                surface,
                owner,
                config,
            },
            device,
            queue,
            pipeline,
            uniform,
            bind_group,
            vertex_buffer,
            depth,
            adapter_name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            format: format!("{format:?}"),
            present: format!("{present:?}"),
        })
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

    fn render(&mut self, frame_index: u64) -> Result<(), String> {
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
                return Err(" surface lost".to_string());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("swapchain validation error".to_string());
            }
        };

        let angle = frame_index as f32 * 0.01;
        let model = glam::Mat4::from_rotation_y(angle)
            * glam::Mat4::from_rotation_x(angle * 0.6);
        let matrix = model.to_cols_array_2d();
        let flat = [
            matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3],
            matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3],
            matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3],
            matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3],
        ];
        self.queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&flat));

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("K3Test frame encoder"),
            },
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("K3Test frame pass"),
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..CUBE_VERTICES.len() as u32, 0..1);
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
    let mut frame_index: u64 = 0;
    let mut fps_frames: u64 = 0;
    let mut last_fps = Instant::now();
    let mut last_status = Instant::now();

    let initial_size = owner.size();
    {
        let mut state = shared.state.lock().expect("engine state mutex");
        if state.shutdown {
            // `owner` is still borrowed here; it is released after this scope.
            drop(owner);
            return;
        }
    }

    match pollster::block_on(Renderer::new(owner, initial_size)) {
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
            log::error!("K3Test bridge initialization failed: {error}");
            return;
        }
    }

    loop {
        let action = {
            let mut state = shared.state.lock().expect("engine state mutex");
            if state.shutdown {
                break;
            }
            while !state.frame_pending && state.resize_pending.is_none() && !state.shutdown {
                state = shared
                    .ready
                    .wait(state)
                    .expect("engine state mutex");
            }
            if state.shutdown {
                break;
            }
            (state.resize_pending.take(), state.frame_pending, state.shutdown)
        };
        shared.frame_requested.store(false, Ordering::Release);

        let (resize, frame_pending, _) = action;
        if let Some((width, height)) = resize {
            if let Some(current) = renderer.as_mut() {
                current.resize(width, height);
                let mut state = shared.state.lock().expect("engine state mutex");
                state.status.size = (width, height);
            }
        }
        if !frame_pending {
            continue;
        }
        frame_index += 1;
        let Some(current) = renderer.as_mut() else {
            continue;
        };
        match current.render(frame_index) {
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
                        "K3Test bridge running backend={} fps={:.1}",
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

    // Drop the GPU resources, then the surface, then the ANativeWindow.
    drop(renderer);
}

fn init_logging() {
    let logger = android_logger::Config::default()
        .with_max_level(log::LevelFilter::Info)
        .with_tag("K3TestBridge");
    android_logger::init_once(logger);
    log::info!("K3Test bridge logging ready");
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
        log::error!("K3Test bridge nativeCreate called with a null Surface");
        return 0;
    }
    let Some(window) = (unsafe { NativeWindow::from_surface(env, surface) }) else {
        log::error!("K3Test bridge could not acquire ANativeWindow");
        return 0;
    };

    let id = NEXT_ENGINE.fetch_add(1, Ordering::Relaxed);
    let shared = Arc::new(Shared::new(id));
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
            log::info!("K3Test bridge engine {id} worker started");
            id as jlong
        }
        Err(error) => {
            engines()
                .lock()
                .expect("engine registry mutex")
                .remove(&id);
            log::error!("K3Test bridge render thread spawn failed: {error}");
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
    log::info!("K3Test bridge engine {id} shutdown requested");
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
    jni_string(
        &mut env,
        &format!(
            "state={} frames={} fps={:.1} adapter={} backend={} format={} present={} size={}x{} detail={}",
            status.state,
            status.frames,
            status.fps,
            status.adapter,
            status.backend,
            status.format,
            status.present,
            status.size.0,
            status.size.1,
            status.detail
        ),
    )
}
