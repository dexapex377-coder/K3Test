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
    external fun nativeOrbit(handle: Long, dx: Float, dy: Float): Boolean
    external fun nativePan(handle: Long, dx: Float, dy: Float): Boolean
    external fun nativeZoom(handle: Long, scale: Float): Boolean
    external fun nativeFrameAll(handle: Long): Boolean
    external fun nativeSetTool(handle: Long, tool: Int): Boolean
    external fun nativePick(handle: Long, x: Float, y: Float): Int
    external fun nativeGizmoHit(handle: Long, x: Float, y: Float): Int
    external fun nativeMoveBegin(handle: Long, x: Float, y: Float): Int
    external fun nativeMoveTo(handle: Long, x: Float, y: Float): Boolean
    external fun nativeMoveEnd(handle: Long): Boolean
    external fun nativeDestroy(handle: Long): Boolean
    external fun nativeInfo(handle: Long): String
}
