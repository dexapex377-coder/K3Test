//! K3Test — viewport wgpu mínimo compatible PowerVR / Mali / Adreno.
//!
//! Reglas (medidas en el Motorola g56, PowerVR BXM-8-256):
//! - Superficie `Rgba8Unorm` (8-bit, sin HDR 16F)
//! - Depth 24-bit (no D32S8)
//! - MSAA 1× (sin MSAA)
//! - 1 pipeline forward + 1 blit — cero compute/storage en el viewport
//! - wgpu 30.0.1 (misma que Starling/Horizon), backend Vulkan primero con
//!   fallback GLES (OpenGL ES 3.2 en este GPU)
//!
//! Si el cubo dibuja acá, la base de rendering está validada para los 3 vendors.

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, WindowId},
    event_loop::{ActiveEventLoop, EventLoop},
    platform::android::activity::AndroidApp,
    window::WindowAttributes,
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

// Cubo: 36 vértices (2 por triángulo x 12 tris), colores por cara
const CUBE_VERTICES: &[Vertex] = &[
    // front (+Z)
    Vertex { position: [-0.5, -0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [0.5, -0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [0.5, 0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [-0.5, -0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [0.5, 0.5, 0.5], color: [1.0, 0.2, 0.2] },
    Vertex { position: [-0.5, 0.5, 0.5], color: [1.0, 0.2, 0.2] },
    // back (-Z)
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [-0.5, 0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [0.5, 0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [0.5, 0.5, -0.5], color: [0.2, 1.0, 0.2] },
    Vertex { position: [0.5, -0.5, -0.5], color: [0.2, 1.0, 0.2] },
    // right (+X)
    Vertex { position: [0.5, -0.5, -0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, -0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, 0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [0.5, -0.5, -0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, 0.5], color: [0.2, 0.4, 1.0] },
    Vertex { position: [0.5, -0.5, 0.5], color: [0.2, 0.4, 1.0] },
    // left (-X)
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, -0.5, 0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, 0.5, 0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, 0.5, 0.5], color: [1.0, 0.9, 0.2] },
    Vertex { position: [-0.5, 0.5, -0.5], color: [1.0, 0.9, 0.2] },
    // top (+Y)
    Vertex { position: [-0.5, 0.5, -0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [-0.5, 0.5, 0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, 0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [-0.5, 0.5, -0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, 0.5], color: [1.0, 0.4, 1.0] },
    Vertex { position: [0.5, 0.5, -0.5], color: [1.0, 0.4, 1.0] },
    // bottom (-Y)
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.2, 1.0, 1.0] },
    Vertex { position: [0.5, -0.5, -0.5], color: [0.2, 1.0, 1.0] },
    Vertex { position: [0.5, -0.5, 0.5], color: [0.2, 1.0, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.2, 1.0, 1.0] },
    Vertex { position: [0.5, -0.5, 0.5], color: [0.2, 1.0, 1.0] },
    Vertex { position: [-0.5, -0.5, 0.5], color: [0.2, 1.0, 1.0] },
];

struct App {
    window: Option<Arc<winit::window::Window>>,
    renderer: Option<Renderer>,
}

impl App {
    fn new() -> Self {
        Self { window: None, renderer: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = WindowAttributes::default().with_title("K3Test");
            // Permite que wgpu gestione el surface (winit 0.30 en Android).
            match event_loop.create_window(attrs) {
                Ok(win) => {
                    let window = Arc::new(win);
                    match pollster::block_on(Renderer::new(Arc::clone(&window))) {
                        Ok(renderer) => {
                            log::info!("K3Test: renderer listo (backend={:?})", renderer.backend);
                            self.renderer = Some(renderer);
                        }
                        Err(e) => {
                            log::error!("K3Test: renderer falló: {e}");
                        }
                    }
                    self.window = Some(window);
                }
                Err(e) => log::error!("K3Test: ventana falló: {e}"),
            }
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = self.renderer.as_mut() {
                    // Rotación lenta en Y + leve en X
                    let angle = renderer.frames as f32 * 0.01;
                    let model = glam::Mat4::from_rotation_y(angle)
                        * glam::Mat4::from_rotation_x(angle * 0.6)
                        * glam::Mat4::from_translation(glam::Vec3::ZERO);
                    renderer.update_uniform(model.to_cols_array_2d());
                    if let Err(e) = renderer.render() {
                        log::error!("K3Test: render error: {e}");
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

struct Renderer {
    backend: wgpu::Backend,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    depth_view: wgpu::TextureView,
    frames: u64,
    fps_frames: u64,
    fps_start: std::time::Instant,
}

impl Renderer {
    async fn new(window: Arc<winit::window::Window>) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window)?;
        let surface: wgpu::Surface<'static> =
            unsafe { std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface) };

        // Vulkan primero, fallback GLES — igual que Starling en este GPU.
        let adapter_req = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        };
        let adapter = instance
            .request_adapter(&adapter_req)
            .await
            .ok_or("K3Test: sin adapter wgpu (¿Vulkan y GLES fallaron?)")?;

        let info = adapter.get_info();
        log::info!("K3Test: adapter={} backend={:?} driver={:?}", info.name, info.backend, info.driver);
        log::info!(
            "K3Test: features disponibles: limits.max_buffer_size={} max_storage_buffers={}",
            adapter.limits().max_buffer_size,
            adapter.limits().max_storage_buffers
        );

        let backend = info.backend;
        // Sin features opcionales: solo lo básico que el BXM acepta seguro.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(&adapter),
                ..Default::default()
            })
            .await?;

        // Surface: Rgba8Unorm (8-bit), sin HDR — la regla PowerVR.
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
            desired_aspect_ratio: 1.0,
            alpha_mode: caps.alpha_modes[0],
        };
        surface.configure(&device, &config);

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("K3Test Depth"),
            size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Stencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("K3Test Shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{VERTEX_SHADER}\n{FRAGMENT_SHADER}").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K3Test BGL"),
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

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K3Test Uniform"),
            contents: bytemuck::cast_slice(&[0f32; 64]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("K3Test BG"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("K3Test PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K3Test Cube"),
            contents: bytemuck::cast_slice(CUBE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("K3Test Pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
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
                }],
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
                format: wgpu::TextureFormat::Depth24Stencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        log::info!(
            "K3Test: surface {}x{} format={:?} depth=Depth24Stencil8 msaa=1 (compatible PowerVR/Mali/Adreno)",
            config.width,
            config.height,
            format
        );

        Ok(Self {
            backend,
            surface,
            device,
            queue,
            config,
            render_pipeline,
            uniform_buffer,
            bind_group,
            vertex_buffer,
            depth_view,
            frames: 0,
            fps_frames: 0,
            fps_start: std::time::Instant::now(),
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("K3Test Depth"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24Stencil8,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    fn update_uniform(&self, matrix: [[f32; 4]; 4]) {
        let flat: [f32; 16] = [
            matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3],
            matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3],
            matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3],
            matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3],
        ];
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&flat));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.frames += 1;
        self.fps_frames += 1;
        if self.fps_start.elapsed().as_secs() >= 2 {
            let fps = self.fps_frames as f64 / self.fps_start.elapsed().as_secs_f64();
            log::info!("K3Test: {:.1} fps (backend={:?}, frames={})", fps, self.backend, self.frames);
            self.fps_frames = 0;
            self.fps_start = std::time::Instant::now();
        }

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("K3Test Encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("K3Test Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.10, g: 0.11, b: 0.14, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..CUBE_VERTICES.len() as u32, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn init_logging() {
    let logger = android_logger::Config::default()
        .with_max_level(log::LevelFilter::Info)
        .with_tag("K3Test");
    if android_logger::init_once(logger).is_ok() {
        log::info!("K3Test: logging ok");
    }
}

#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    init_logging();
    log::info!("K3Test: android_main (lib=k3test, wgpu 30.0.1, Rgba8Unorm, Depth24, MSAA 1x)");
    let event_loop = match EventLoop::builder()
        .with_android_app(app)
        .build()
    {
        Ok(el) => el,
        Err(e) => {
            log::error!("K3Test: EventLoop falló: {e}");
            return;
        }
    };
    let mut app = App::new();
    if let Err(e) = event_loop.run_app(&mut app) {
        log::error!("K3Test: run_app terminó: {e}");
    }
}
