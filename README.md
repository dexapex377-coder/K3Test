# K3Test

**Spike Compose + JNI + wgpu 30.0.1** for validar ANativeWindow, SurfaceView y wgpu en un Motorola g56 (MediaTek mt6855, PowerVR BXM-8-256).

> Nombre provisorio. No es Blender. Cuando el DCC esté estable con las tools y modificadores, se renombra a *Kronos3D Evolution*.

## Qué demuestra (y qué no)

- `MainActivity` Compose (ComponentActivity, sin NativeActivity) hostea un SurfaceView vía AndroidView.
- Kotlin pasa `android.view.Surface` por JNI a Rust.
- Rust convierte a `ANativeWindow` con `ANativeWindow_fromSurface` (incrementa referencia nativa).
- El render vive en un hilo nativo dedicado con `wgpu::SurfaceTarget::Window(Box<dyn WindowHandle>)`.
- Verificable: create → present → resize → destroy de la SurfaceView sin bloquear el UI thread ni liberar dos veces.
- Reglas PowerVR del spike: superficie sRGB 8-bit, depth 24-bit, MSAA 1×, un solo render pass, sin compute/storage/HDR 16F.

## Build (CI)

GitHub Actions compila `libk3test.so` para `aarch64-linux-android` (Rust 1.9x) y empaqueta con Gradle.

## Probar en el Motorola

```bash
adb -s <IP:PUERTO> install -r app-debug.apk
adb -s <IP:PUERTO> shell monkey -p dev.k3test -c android.intent.category.LAUNCHER 1
adb -s <IP:PUERTO> logcat -s K3TestBridge:V
```

El Compose muestra estado de la sesión nativa y frames presentados. La rotación del cubo continua con Choreographer (cada vsync) a través de `nativeFrame`.
