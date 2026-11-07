package dev.k3test

import android.content.Context
import android.view.SurfaceHolder
import android.view.SurfaceView

/**
 * SurfaceView host. Kotlin owns creation and destruction of the Java Surface;
 * Rust owns the ANativeWindow and wgpu surface after nativeCreate.
 */
internal class ViewportSurface(
    context: Context,
    private val onHandleChanged: (Long) -> Unit
) : SurfaceView(context), SurfaceHolder.Callback2 {

    private var nativeHandle: Long = 0L
    private var surfaceAlive: Boolean = false

    init {
        holder.addCallback(this)
        holder.setFormat(android.graphics.PixelFormat.RGBA_8888)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        surfaceAlive = true
        val newHandle = NativeBridge.nativeCreate(holder.surface)
        nativeHandle = newHandle
        onHandleChanged(newHandle)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        val handle = nativeHandle
        if (surfaceAlive && handle != 0L) {
            NativeBridge.nativeResize(handle, width, height)
        }
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        surfaceAlive = false
        val handleToDestroy = nativeHandle
        nativeHandle = 0L
        onHandleChanged(0L)
        if (handleToDestroy != 0L) {
            // Non-blocking: Rust signals its render worker and releases
            // ANativeWindow/wgpu resources on that worker thread.
            NativeBridge.nativeDestroy(handleToDestroy)
        }
    }

    override fun surfaceRedrawNeeded(holder: SurfaceHolder) {
        val handle = nativeHandle
        if (surfaceAlive && handle != 0L) {
            NativeBridge.nativeFrame(handle)
        }
    }

    override fun surfaceRedrawNeededAsync(
        holder: SurfaceHolder,
        drawingFinished: Runnable
    ) {
        val handle = nativeHandle
        if (surfaceAlive && handle != 0L) {
            NativeBridge.nativeFrame(handle)
        }
        drawingFinished.run()
    }

    override fun onDetachedFromWindow() {
        val handleToDestroy = nativeHandle
        nativeHandle = 0L
        surfaceAlive = false
        onHandleChanged(0L)
        if (handleToDestroy != 0L) {
            NativeBridge.nativeDestroy(handleToDestroy)
        }
        super.onDetachedFromWindow()
    }
}
