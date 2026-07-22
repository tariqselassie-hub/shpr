//! Portable 4-Way Unrolled Scalar Fallback Engine for Continuous Phase Manifolds.

use std::f64::consts::PI;

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

    /// Normalizes a phase angle to the principal interval $[-\pi, \pi]$.
    #[inline]
    pub fn normalize_angle(angle: f64) -> f64 {
        let mut a = (angle + PI) % (2.0 * PI);
        if a < 0.0 {
            a += 2.0 * PI;
        }
        a - PI
    }

    /// In-place 4-way loop unrolled scalar phase addition.
    pub fn add_phases_scalar(&mut self, rhs: &ScalarPhaseVector) {
        let dim = self.angles.len();
        let chunks = dim / 4;
        let remainder = dim % 4;

        for c in 0..chunks {
            let idx = c * 4;
            self.angles[idx] = Self::normalize_angle(self.angles[idx] + rhs.angles[idx]);
            self.angles[idx + 1] = Self::normalize_angle(self.angles[idx + 1] + rhs.angles[idx + 1]);
            self.angles[idx + 2] = Self::normalize_angle(self.angles[idx + 2] + rhs.angles[idx + 2]);
            self.angles[idx + 3] = Self::normalize_angle(self.angles[idx + 3] + rhs.angles[idx + 3]);
        }

        for i in (dim - remainder)..dim {
            self.angles[i] = Self::normalize_angle(self.angles[i] + rhs.angles[i]);
        }
    }
}
