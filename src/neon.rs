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

    /// Deterministically generates a continuous phase vector from seed and string key.
    pub fn from_seed(seed: u64, key: &str) -> Self {
        use crate::phase_math::{mix_str_key_into_state, next_splitmix64};
        use std::f32::consts::PI;

        let mut state = mix_str_key_into_state(seed, key);
        let mut vec = Self::new();
        for i in 0..MANIFOLD_DIM {
            let h = next_splitmix64(&mut state);
            let norm = (h as f32) / (u64::MAX as f32);
            vec.angles[i] = norm * 2.0 * PI - PI;
        }
        vec
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

    /// Vectorized Minimax Taylor Cosine Resonance Score using ARM NEON.
    ///
    /// Computes phase alignment score $\frac{1}{D} \sum \cos(\theta_{1,i} - \theta_{2,i})$.
    ///
    /// # Safety
    /// Requires `aarch64` target architecture with NEON support.
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn phase_resonance_neon(a: &NEONPhaseVector, b: &NEONPhaseVector) -> f32 {
        let mut acc = vdupq_n_f32(0.0);

        let two_pi = vdupq_n_f32(std::f32::consts::TAU);
        let inv_two_pi = vdupq_n_f32(1.0 / std::f32::consts::TAU);

        let c2 = vdupq_n_f32(-0.49999999);
        let c4 = vdupq_n_f32(0.04166666);
        let one = vdupq_n_f32(1.0);

        for i in (0..MANIFOLD_DIM).step_by(4) {
            let v1 = vld1q_f32(a.angles.as_ptr().add(i));
            let v2 = vld1q_f32(b.angles.as_ptr().add(i));

            let diff = vsubq_f32(v1, v2);

            let k = vrndnq_f32(vmulq_f32(diff, inv_two_pi));
            let wrapped_diff = vfmsq_f32(diff, k, two_pi);

            let diff_sq = vmulq_f32(wrapped_diff, wrapped_diff);

            let poly = vfmaq_f32(diff_sq, c4, c2);
            let cos_approx = vfmaq_f32(diff_sq, poly, one);

            acc = vaddq_f32(acc, cos_approx);
        }

        let mut temp = [0.0f32; 4];
        vst1q_f32(temp.as_mut_ptr(), acc);
        let total_sum: f32 = temp.iter().sum();
        total_sum / (MANIFOLD_DIM as f32)
    }
}
