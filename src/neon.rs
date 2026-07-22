//! ARM64 NEON Hardware Acceleration Engine for T^2048 Continuous Phase Manifold.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Dimension of the NEON hardware-accelerated phase manifold $D = 2048$.
pub const MANIFOLD_DIM: usize = 2048;

/// Memory-aligned 128-bit ARM NEON Phase Vector ($D = 2048$).
///
/// Guaranteed 16-byte alignment for aligned `vld1q_f32` / `vst1q_f32` vector operations.
#[repr(C, align(16))]
#[derive(Clone, Debug)]
pub struct NEONPhaseVector {
    /// Array of single-precision float phase angles in radians.
    pub angles: [f32; MANIFOLD_DIM],
}

impl Default for NEONPhaseVector {
    fn default() -> Self {
        Self::new()
    }
}

impl NEONPhaseVector {
    /// Creates a zero-initialized NEON phase vector.
    pub fn new() -> Self {
        Self {
            angles: [0.0; MANIFOLD_DIM],
        }
    }

    /// Vectorized Phase Addition over $\mathbb{T}^{2048}$ using ARM NEON (4 x f32 per lane).
    ///
    /// # Safety
    /// Requires `aarch64` target architecture with NEON support.
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn add_phases_neon(&mut self, rhs: &NEONPhaseVector) {
        let two_pi = vdupq_n_f32(std::f32::consts::TAU);
        let inv_two_pi = vdupq_n_f32(1.0 / std::f32::consts::TAU);

        for i in (0..MANIFOLD_DIM).step_by(4) {
            let p1_ptr = self.angles.as_mut_ptr().add(i);
            let p2_ptr = rhs.angles.as_ptr().add(i);

            let v1 = vld1q_f32(p1_ptr);
            let v2 = vld1q_f32(p2_ptr);

            let sum = vaddq_f32(v1, v2);

            let k = vrndnq_f32(vmulq_f32(sum, inv_two_pi));
            let wrapped = vsubq_f32(sum, vmulq_f32(k, two_pi));

            vst1q_f32(p1_ptr, wrapped);
        }
    }
}
