package dev.k3test

import android.content.Context
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView

/**
 * Touch-aware SurfaceView host.
 *
 * One active pointer orbits the camera. Two active pointers pan from the
 * centroid and pinch to zoom. Exact commands are constructed in Rust; this
 * class only translates standard Android pointer updates.
 */
internal class ViewportSurface(
    context: Context,
    private val onHandleChanged: (Long) -> Unit
) : SurfaceView(context), SurfaceHolder.Callback2 {

    private var nativeHandle: Long = 0L
    private var surfaceAlive: Boolean = false

    private companion object {
        const val MAX_POINTERS = 8
    }

    private val pointerIds = IntArray(MAX_POINTERS) { MotionEvent.INVALID_POINTER_ID }
    private val pointerX = FloatArray(MAX_POINTERS)
    private val pointerY = FloatArray(MAX_POINTERS)
    private var singleId = MotionEvent.INVALID_POINTER_ID
    private var lastSingleX = 0f
    private var lastSingleY = 0f
    private var lastCentroid: Pair<Float, Float>? = null
    private var lastSpread = 0f

    init {
        isFocusable = true
        isFocusableInTouchMode = true
        isClickable = true
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

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val handle = nativeHandle
        if (!surfaceAlive || handle == 0L) {
            return false
        }

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                val index = event.actionIndex
                setPointer(event.getPointerId(index), event.getX(index), event.getY(index))
                resetGestureState()
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                val index = event.actionIndex
                setPointer(event.getPointerId(index), event.getX(index), event.getY(index))
                resetGestureState()
            }
            MotionEvent.ACTION_MOVE -> {
                for (index in 0 until event.pointerCount) {
                    setPointer(event.getPointerId(index), event.getX(index), event.getY(index))
                }
                handleMove(handle)
            }
            MotionEvent.ACTION_POINTER_UP -> {
                val id = event.getPointerId(event.actionIndex)
                clearPointer(id)
                resetGestureState()
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                clearAllPointers()
                resetGestureState()
            }
        }
        return true
    }

    private fun handleMove(handle: Long) {
        val active = activePointers()
        if (active.size == 1) {
            val (id, point) = active[0]
            if (singleId != id) {
                singleId = id
                lastSingleX = point.first
                lastSingleY = point.second
                return
            }
            val dx = point.first - lastSingleX
            val dy = point.second - lastSingleY
            lastSingleX = point.first
            lastSingleY = point.second
            if (dx != 0f || dy != 0f) {
                NativeBridge.nativeOrbit(handle, dx, dy)
            }
            return
        }

        if (active.size < 2) return
        val first = active[0].second
        val second = active[1].second
        val centroid = ((first.first + second.first) * 0.5f) to ((first.second + second.second) * 0.5f)
        val spread = hypot(second.first - first.first, second.second - first.second)
        val previousCentroid = lastCentroid
        val previousSpread = lastSpread
        if (previousCentroid != null && previousSpread > 0f && spread > 0f) {
            NativeBridge.nativePan(
                handle,
                centroid.first - previousCentroid.first,
                centroid.second - previousCentroid.second
            )
            NativeBridge.nativeZoom(handle, spread / previousSpread)
        }
        lastCentroid = centroid
        lastSpread = spread
    }

    private fun setPointer(id: Int, x: Float, y: Float) {
        val existing = pointerIds.indexOf(id)
        val slot = if (existing >= 0) existing else pointerIds.indexOfFirst { it == MotionEvent.INVALID_POINTER_ID }
        if (slot < 0) return
        pointerIds[slot] = id
        pointerX[slot] = x
        pointerY[slot] = y
    }

    private fun clearPointer(id: Int) {
        val slot = pointerIds.indexOf(id)
        if (slot >= 0) {
            pointerIds[slot] = MotionEvent.INVALID_POINTER_ID
        }
    }

    private fun clearAllPointers() {
        pointerIds.fill(MotionEvent.INVALID_POINTER_ID)
    }

    private fun activePointers(): List<Pair<Int, Pair<Float, Float>>> {
        val result = mutableListOf<Pair<Int, Pair<Float, Float>>>()
        for (i in pointerIds.indices) {
            val id = pointerIds[i]
            if (id != MotionEvent.INVALID_POINTER_ID) {
                result += id to (pointerX[i] to pointerY[i])
            }
        }
        return result
    }

    private fun resetGestureState() {
        singleId = MotionEvent.INVALID_POINTER_ID
        lastSingleX = 0f
        lastSingleY = 0f
        lastCentroid = null
        lastSpread = 0f
    }

    private fun hypot(x: Float, y: Float): Float {
        return kotlin.math.sqrt(x * x + y * y)
    }
}
