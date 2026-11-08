package dev.k3test

import android.view.Choreographer
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.displayCutoutPadding
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
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
    var tool by remember { mutableStateOf(1) }

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
                .displayCutoutPadding()
                .statusBarsPadding()
                .padding(start = 4.dp, top = 176.dp),
            tool = tool,
            onToolChange = { newTool ->
                tool = newTool
                if (handle != 0L) {
                    NativeBridge.nativeSetTool(handle, tool)
                }
            }
        )
    }

    LaunchedEffect(handle) {
        // Self-healing sync: re-applying the tool is a no-op in Rust when
        // nothing changed, so polling keeps the native side consistent
        // across surface recreations at zero idle cost.
        while (true) {
            if (handle != 0L) {
                NativeBridge.nativeSetTool(handle, tool)
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
 * Left vertical tool bar (top-left, safe-area aware). Ships Move (Figcraft
 * vector) and Rotate (temporary default icon until the Figcraft export).
 */
@Composable
private fun ToolBar(
    modifier: Modifier = Modifier,
    tool: Int,
    onToolChange: (Int) -> Unit,
) {
    Column(modifier = modifier) {
        ToolButton(active = tool == 1, onClick = { onToolChange(if (tool == 1) 0 else 1) }) {
            MoveToolIcon()
        }
        ToolButton(active = tool == 2, onClick = { onToolChange(if (tool == 2) 0 else 2) }) {
            RotateToolIcon()
        }
    }
}

@Composable
private fun ToolButton(
    active: Boolean,
    onClick: () -> Unit,
    content: @Composable () -> Unit,
) {
    Box(
        modifier = Modifier
            .padding(bottom = 8.dp)
            .size(68.dp)
            .clickable(onClick = onClick),
        contentAlignment = androidx.compose.ui.Alignment.Center
    ) {
        content()
        if (active) {
            // Blue active outline hugging the icon bounds.
            Box(
                modifier = Modifier
                    .size(64.dp, 52.dp)
                    .border(2.dp, Color(0xFF4A9BE8), RoundedCornerShape(14.dp))
            )
        }
    }
}

@Composable
private fun MoveToolIcon() {
    // Figcraft vector: rounded black background + white arrows.
    // The drawable owns its background; the toolbar adds none.
    Image(
        painter = painterResource(id = R.drawable.move),
        contentDescription = "Move tool",
        contentScale = ContentScale.FillBounds,
        modifier = Modifier.size(64.dp, 52.dp)
    )
}

/** Temporary default Rotate icon (double curved arrow) pending Figcraft export. */
@Composable
private fun RotateToolIcon() {
    Canvas(modifier = Modifier.size(64.dp, 52.dp)) {
        drawArc(
            color = Color(0xFFD9D9D9),
            startAngle = 20f,
            sweepAngle = 240f,
            useCenter = false,
            style = androidx.compose.ui.graphics.drawscope.Stroke(
                width = 6f,
                cap = androidx.compose.ui.graphics.StrokeCap.Round
            )
        )
        drawArc(
            color = Color(0xFFD9D9D9),
            startAngle = 200f,
            sweepAngle = 240f,
            useCenter = false,
            style = androidx.compose.ui.graphics.drawscope.Stroke(
                width = 6f,
                cap = androidx.compose.ui.graphics.StrokeCap.Round
            )
        )
    }
}
