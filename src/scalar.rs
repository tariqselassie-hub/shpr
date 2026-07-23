//! Portable 4-Way Unrolled Scalar Fallback Engine for Continuous Phase Manifolds.

use std::f64::consts::PI;

use crate::phase_math::{mix_str_key_into_state, next_splitmix64};

/// Default manifold dimension $D = 2048$.
pub const MANIFOLD_DIM: usize = 2048;

/// Portable unrolled scalar fallback phase vector.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarPhaseVector {
    /// Vector of continuous phase angles in radians.
    pub angles: Vec<f64>,
}

impl ScalarPhaseVector {
    /// Creates a zero-initialized scalar phase vector of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            angles: vec![0.0; dim],
        }
    }

    /// Creates a zero-initialized scalar phase vector of dimension `dim`.
    /// # Example
    /// ```rust
    /// use shpr::ScalarPhaseVector;
    /// let vec = ScalarPhaseVector::zeros(512);
    /// assert_eq!(vec.angles.len(), 512);
    /// ```
    pub fn zeros(dim: usize) -> Self {
        Self {
            angles: vec![0.0; dim],
        }
    }

    /// Normalizes a phase angle to the principal interval $[-\pi, \pi]$.
    #[inline]
    pub fn normalize_angle(angle: f64) -> f64 {
        crate::phase_math::normalize_angle(angle)
    }

    /// Deterministically generates a phase vector from a seed and string key.
    pub fn from_seed(seed: u64, key: &str, dim: usize) -> Self {
        let mut state = mix_str_key_into_state(seed, key);
        let mut angles = Vec::with_capacity(dim);
        for _ in 0..dim {
            let h = next_splitmix64(&mut state);
            let norm = (h as f64) / (u64::MAX as f64);
            angles.push(norm * 2.0 * PI - PI);
        }
        Self { angles }
    }

    /// In-place 4-way loop unrolled scalar phase addition.
    pub fn add_phases_scalar(&mut self, rhs: &ScalarPhaseVector) {
        let dim = self.angles.len();
        let chunks = dim / 4;
        let remainder = dim % 4;

        for c in 0..chunks {
            let idx = c * 4;
            self.angles[idx] = Self::normalize_angle(self.angles[idx] + rhs.angles[idx]);
            self.angles[idx + 1] =
                Self::normalize_angle(self.angles[idx + 1] + rhs.angles[idx + 1]);
            self.angles[idx + 2] =
                Self::normalize_angle(self.angles[idx + 2] + rhs.angles[idx + 2]);
            self.angles[idx + 3] =
                Self::normalize_angle(self.angles[idx + 3] + rhs.angles[idx + 3]);
        }

        for i in (dim - remainder)..dim {
            self.angles[i] = Self::normalize_angle(self.angles[i] + rhs.angles[i]);
        }
    }
}
