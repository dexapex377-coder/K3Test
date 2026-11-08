package dev.k3test

import android.view.Choreographer
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicText
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            ViewportScreen()
        }
    }
}

@Composable
private fun ViewportScreen() {
    var handle by remember { mutableStateOf(0L) }
    var status by remember { mutableStateOf("starting") }
    var autoFrame by remember { mutableStateOf(false) }
    var moveActive by remember { mutableStateOf(true) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF1A1C24))
    ) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { context ->
                ViewportSurface(context) { newHandle ->
                    handle = newHandle
                }
            }
        )

        Column(
            modifier = Modifier
                .statusBarsPadding()
                .padding(start = 12.dp, top = 12.dp, end = 12.dp)
        ) {
            BasicText(text = "K3Test Phase 1.0 viewport", style = TextStyle(color = Color.White))
            BasicText(text = "1-finger orbit  -  2-finger pan/pinch", style = TextStyle(color = Color.White))
            BasicText(text = status, style = TextStyle(color = Color.White))
            Row(modifier = Modifier.padding(top = 8.dp)) {
                BasicText(
                    text = if (autoFrame) "Auto frame: on" else "Auto frame: off",
                    style = TextStyle(color = Color.Cyan),
                    modifier = Modifier.clickable { autoFrame = !autoFrame }
                )
                BasicText(
                    text = "  Frame now",
                    style = TextStyle(color = Color.Cyan),
                    modifier = Modifier.clickable {
                        if (handle != 0L) NativeBridge.nativeFrame(handle)
                    }
                )
                BasicText(
                    text = "  Frame all",
                    style = TextStyle(color = Color.Cyan),
                    modifier = Modifier.clickable {
                        if (handle != 0L) NativeBridge.nativeFrameAll(handle)
                    }
                )
            }
        }

        ToolBar(
            modifier = Modifier
                .statusBarsPadding()
                .padding(start = 4.dp, top = 176.dp),
            moveActive = moveActive,
            onMoveToggle = {
                moveActive = !moveActive
                if (handle != 0L) {
                    NativeBridge.nativeSetTool(handle, if (moveActive) 1 else 0)
                }
            }
        )
    }

    LaunchedEffect(handle) {
        if (handle != 0L) {
            NativeBridge.nativeSetTool(handle, if (moveActive) 1 else 0)
        }
        while (true) {
            if (handle != 0L) {
                status = NativeBridge.nativeInfo(handle)
            }
            delay(500)
        }
    }

    val choreographer = remember { Choreographer.getInstance() }
    val frameCallback = remember(handle, autoFrame) {
        if (!autoFrame || handle == 0L) {
            null
        } else {
            object : Choreographer.FrameCallback {
                override fun doFrame(frameTimeNanos: Long) {
                    if (handle != 0L) {
                        NativeBridge.nativeFrame(handle)
                    }
                    if (autoFrame) {
                        choreographer.postFrameCallback(this)
                    }
                }
            }
        }
    }

    DisposableEffect(frameCallback) {
        if (frameCallback != null) {
            choreographer.postFrameCallback(frameCallback)
        }
        onDispose {
            if (frameCallback != null) {
                choreographer.removeFrameCallback(frameCallback)
            }
        }
    }
}

/**
 * Left vertical tool bar (top-left, safe-area aware). Only Move ships in
 * this phase; slots for Rotate/Scale stay reserved in the layout but are
 * not drawn as unfinished features.
 */
@Composable
private fun ToolBar(
    modifier: Modifier = Modifier,
    moveActive: Boolean,
    onMoveToggle: () -> Unit,
) {
    Column(
        modifier = modifier
            .padding(4.dp)
            .background(Color(0xFF232732), RoundedCornerShape(16.dp))
            .padding(4.dp)
    ) {
        Box(
            modifier = Modifier
                .size(56.dp)
                .background(
                    if (moveActive) Color(0xFF3A5068) else Color(0xFF2C3038),
                    RoundedCornerShape(12.dp)
                )
                .border(
                    1.dp,
                    if (moveActive) Color(0xFF6FB6E8) else Color.Transparent,
                    RoundedCornerShape(12.dp)
                )
                .clickable(onClick = onMoveToggle),
            contentAlignment = androidx.compose.ui.Alignment.Center
        ) {
            MoveToolIcon()
        }
    }
}

/** Blender-style Move icon: X (red), Y (green), Z (blue) arrows. */
@Composable
private fun MoveToolIcon() {
    Canvas(modifier = Modifier.size(40.dp)) {
        val center = Offset(size.width / 2f, size.height / 2f)
        val length = size.minDimension * 0.36f

        fun arrow(direction: Offset, color: Color) {
            val tip = center + direction * length
            drawLine(color, center, tip, strokeWidth = 4f, cap = StrokeCap.Round)
            // Back-folded barb perpendicular to the arrow direction.
            val back = tip - direction * length * 0.32f
            val perp = Offset(-direction.y, direction.x) * length * 0.16f
            drawLine(color, tip, back + perp, strokeWidth = 4f, cap = StrokeCap.Round)
            drawLine(color, tip, back - perp, strokeWidth = 4f, cap = StrokeCap.Round)
        }

        arrow(Offset(1f, 0f), Color(0xFFFF4D4D))
        arrow(Offset(-0.55f, -0.84f), Color(0xFF59E65C))
        arrow(Offset(-0.55f, 0.84f), Color(0xFF5A8CFF))
        drawCircle(Color.White, radius = 3f, center = center)
    }
}
