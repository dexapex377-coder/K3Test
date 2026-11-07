# K3Test

**Viewport wgpu mínimo** para validar rendering en GPUs móviles (PowerVR / Mali / Adreno).

> Nombre provisorio. No es Blender. Cuando el DCC esté estable con las tools y modificadores, se renombra a *Kronos3D Evolution*.

## Qué es

- **Motor:** Rust, `wgpu 30.0.1` (misma que Starling/Horizon) + `winit 0.30` con `android-native-activity`.
- **Viewport:** cubo rotando con colores por cara, 1 pipeline forward.
- **Formato:** `Rgba8Unorm` + `Depth24Stencil8`, **MSAA 1×**, sin HDR, sin compute/storage en el viewport.

Estas reglas vienen de medir el hardware real:

- Motorola g56 5G (MediaTek mt6855, **PowerVR B-Series BXM-8-256**, Vulkan **1.3.1** driver 25.1@6715691).
- Starling (`dev.horizon.android`) corre Vulkan con `Rgba8Unorm/Mailbox` y features limitadas.
- TextureCraft pide `Rgba16Float` HDR + compute — eso es lo que no dibuja en este PowerVR.

## Build (CI)

```bash
# GitHub Actions compila y sube k3test-debug-apk
# Motorola solo instala/prueba por adb
```

## Probar en el Motorola

```bash
adb -s <IP:PUERTO> install -r app-debug.apk
adb -s <IP:PUERTO> shell monkey -p dev.k3test -c android.intent.category.LAUNCHER 1
adb -s <IP:PUERTO> logcat -s K3Test:V
# Esperado: "adapter=... backend=... " + cubo girando + "fps" cada 2s
```

## Plan

- [x] Fase 0: viewport mínimo (este repo) — validar PowerVR.
- [ ] Fase 1: mesh viewer + transform + edit mode.
- [ ] Fase 2: modificadores.
- [ ] Fase 3: texture paint (crates `texel/compose/paint/bake` de TextureCraft).
- [ ] Fase 4: animación. Fase 5: sculpt.
