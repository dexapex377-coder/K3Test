package dev.k3test

import android.view.Surface

/**
 * Rust owns ANativeWindow/wgpu after nativeCreate. Kotlin only stores an opaque
 * integer engine ID for lifetime management.
 */
internal object NativeBridge {
    init {
        System.loadLibrary("k3test")
    }

    external fun nativeCreate(surface: Surface): Long
    external fun nativeResize(handle: Long, width: Int, height: Int): Boolean
    external fun nativeFrame(handle: Long): Boolean
    external fun nativeDestroy(handle: Long): Boolean
    external fun nativeInfo(handle: Long): String
}
