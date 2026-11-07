package dev.k3test;

/** K3Test — viewport wgpu mínimo. Solo carga la lib y deja todo a Rust. */
public final class K3Activity extends android.app.NativeActivity {
    static { System.loadLibrary("k3test"); }
}
