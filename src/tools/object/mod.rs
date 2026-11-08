//! Object Mode tools (translate now, scale later).
//!
//! Mirrors Blender's `transform_gizmo_3d.cc`: each manipulator binds to
//! `TRANSFORM_OT_*` with a single-axis `constraint_axis`, and the
//! active tool gates which gizmo set is drawn
//! (`V3D_GIZMO_SHOW_OBJECT_*`).

pub mod rotate;
pub mod translate;
