//! AVX2 & FMA Hardware Acceleration Engine for T^2048 Continuous Phase Manifold

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const MANIFOLD_DIM: usize = 2048;

/// Memory-aligned 256-bit AVX2 Hardware Accelerated Phase Vector
#[repr(C, align(32))]
#[derive(Clone, Debug)]
pub struct AVX2PhaseVector {
    pub angles: [f32; MANIFOLD_DIM],
}

impl AVX2PhaseVector {
    pub fn new() -> Self {
        Self {
            angles: [0.0; MANIFOLD_DIM],
        }
    }

    pub fn from_f64_slice(slice: &[f64]) -> Self {
        let mut vec = Self::new();
        for (i, &v) in slice.iter().take(MANIFOLD_DIM).enumerate() {
            vec.angles[i] = v as f32;
        }
        vec
    }

    pub fn from_seed(seed: u64, key: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vec = Self::new();
        for i in 0..MANIFOLD_DIM {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            key.hash(&mut hasher);
            i.hash(&mut hasher);
            let h = hasher.finish();
            let norm = (h as f32) / (u64::MAX as f32);
            vec.angles[i] = norm * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
        }
        vec
    }

    /// Vectorized Phase Addition over T^2048 using AVX2 & FMA (8 x f32 per lane)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn add_phases_avx2(&mut self, rhs: &AVX2PhaseVector) {
        let two_pi = _mm256_set1_ps(std::f32::consts::TAU);
        let inv_two_pi = _mm256_set1_ps(1.0 / std::f32::consts::TAU);

        for i in (0..MANIFOLD_DIM).step_by(8) {
            let p1_ptr = self.angles.as_mut_ptr().add(i);
            let p2_ptr = rhs.angles.as_ptr().add(i);

            let v1 = _mm256_load_ps(p1_ptr);
            let v2 = _mm256_load_ps(p2_ptr);

            let sum = _mm256_add_ps(v1, v2);

            let k = _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(_mm256_mul_ps(sum, inv_two_pi));
            let wrapped = _mm256_fnmadd_ps(k, two_pi, sum);

            _mm256_store_ps(p1_ptr, wrapped);
        }
    }

    /// Vectorized Minimax Taylor Cosine Resonance Score using AVX2 & FMA
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn phase_resonance_avx2(a: &AVX2PhaseVector, b: &AVX2PhaseVector) -> f32 {
        let mut acc = _mm256_setzero_ps();

        let two_pi = _mm256_set1_ps(std::f32::consts::TAU);
        let inv_two_pi = _mm256_set1_ps(1.0 / std::f32::consts::TAU);

        let c2 = _mm256_set1_ps(-0.49999999);
        let c4 = _mm256_set1_ps(0.04166666);
        let one = _mm256_set1_ps(1.0);

        for i in (0..MANIFOLD_DIM).step_by(8) {
            let v1 = _mm256_load_ps(a.angles.as_ptr().add(i));
            let v2 = _mm256_load_ps(b.angles.as_ptr().add(i));

            let diff = _mm256_sub_ps(v1, v2);

            let k = _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(_mm256_mul_ps(diff, inv_two_pi));
            let wrapped_diff = _mm256_fnmadd_ps(k, two_pi, diff);

            let diff_sq = _mm256_mul_ps(wrapped_diff, wrapped_diff);

            let poly = _mm256_fmadd_ps(diff_sq, c4, c2);
            let cos_approx = _mm256_fmadd_ps(diff_sq, poly, one);

            acc = _mm256_add_ps(acc, cos_approx);
        }

        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), acc);

        let total_sum: f32 = temp.iter().sum();
        total_sum / (MANIFOLD_DIM as f32)
    }
}
