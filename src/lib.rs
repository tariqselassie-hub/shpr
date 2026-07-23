//! # `shpr`: Senojian-Hyperdimensional Phase-Resonant Graph Attention Engine
//!
//! A zero-dependency, ultra-lightweight, hardware-accelerated Vector Symbolic Architecture (VSA)
//! and Hyperdimensional Computing (HDC) phase memory engine in Rust.
//!
//! ## Overview
//!
//! `shpr` solves the $O(N^2)$ memory and computation bottleneck of Transformer Attention by projecting
//! feature streams onto continuous toroidal phase manifolds ($\mathbb{T}^D = S^1 \times \dots \times S^1$).
//!
//! Key advantages include:
//! - **$O(1)$ Constant Memory:** Sequence context is accumulated into complex phasor state vectors.
//! - **Deterministic Unbinding:** Continuous phase unbinding retrieves exact bound values ($\text{SNR} = \infty$).
//! - **Zero Matrix Multiplications:** Softmax and pairwise dot products are replaced with phase resonance alignment $\sum \cos(\Delta\theta)$.
//! - **Hardware Acceleration:** Accelerated using AVX2 & FMA (x86_64) or NEON (ARM64) SIMD intrinsics.
//!
//! ## Quickstart
//!
//! ```rust
//! use shpr::{SenojianPhaseVector, SHPRGraphAttention};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let dim = 1024;
//!
//!     // Create phase vectors for key and value
//!     let key = SenojianPhaseVector::from_seed(42, "user_id", dim);
//!     let val = SenojianPhaseVector::from_seed(42, "user_name", dim);
//!
//!     // Continuous Toroidal Binding: (theta_k + theta_v) mod 2PI
//!     let bound = key.bind(&val, 0.0)?;
//!
//!     // Lossless Unbinding: (theta_bound - theta_k) mod 2PI
//!     let retrieved = bound.unbind(&key, 0.0)?;
//!
//!     // Verify exact phase resonance match (1.0 = exact match)
//!     let score = retrieved.resonance(&val);
//!     assert!((score - 1.0).abs() < 1e-9);
//!
//!     Ok(())
//! }
//! ```

pub mod avx2;
pub mod neon;
pub mod scalar;

pub use avx2::AVX2PhaseVector;
pub use scalar::ScalarPhaseVector;

use std::f64::consts::PI;

mod phase_math;

use phase_math::{mix_str_key_into_state, next_splitmix64, normalize_angle};

/// Errors emitted by `shpr` operations.
#[derive(Debug, Clone)]
pub enum ShprError {
    /// Dimension mismatch between two vectors or between a vector and an attention engine state.
    DimMismatch {
        /// Expected vector dimension.
        expected: usize,
        /// Provided vector dimension.
        got: usize,
    },
}

impl std::fmt::Display for ShprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShprError::DimMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for ShprError {}

/// Continuous Senojian Phase Vector on a $D$-dimensional torus manifold $\mathbb{T}^D$.
///
/// Phase angles are represented in radians within the normalized continuous interval $[-\pi, \pi]$.
#[derive(Clone, Debug, PartialEq)]
pub struct SenojianPhaseVector {
    /// Internal array of phase angles in radians.
    pub phases: Vec<f64>,
}

impl SenojianPhaseVector {
    /// Creates a zero-initialized phase vector of dimension `dim`.
    ///
    /// # Example
    /// ```rust
    /// use shpr::SenojianPhaseVector;
    /// let vec = SenojianPhaseVector::zeros(512);
    /// assert_eq!(vec.dim(), 512);
    /// ```
    pub fn zeros(dim: usize) -> Self {
        Self {
            phases: vec![0.0; dim],
        }
    }

    /// Returns the dimension $D$ of the phase vector manifold.
    #[inline]
    pub fn dim(&self) -> usize {
        self.phases.len()
    }

    /// Creates a phase vector from arbitrary real features by applying $\tanh(x) \cdot \pi$.
    ///
    /// # Example
    /// ```rust
    /// use shpr::SenojianPhaseVector;
    /// let vec = SenojianPhaseVector::from_features(&[0.5, -1.2, 3.0]);
    /// assert_eq!(vec.dim(), 3);
    /// ```
    pub fn from_features(features: &[f64]) -> Self {
        let phases = features.iter().map(|&x| x.tanh() * PI).collect();
        Self { phases }
    }

    /// Deterministically generates a quasi-orthogonal continuous phase vector from a seed and string key.
    pub fn from_seed(seed: u64, key: &str, dim: usize) -> Self {
        let mut state = mix_str_key_into_state(seed, key);
        let mut phases = Vec::with_capacity(dim);
        for _ in 0..dim {
            let h = next_splitmix64(&mut state);
            let norm = (h as f64) / (u64::MAX as f64);
            phases.push(norm * 2.0 * PI - PI);
        }
        Self { phases }
    }

    /// Performs continuous toroidal binding ($\theta_A + \theta_B + \Delta\phi \pmod{2\pi}$).
    ///
    /// # Errors
    /// Returns [`ShprError::DimMismatch`] if `self.dim() != rhs.dim()`.
    ///
    /// # Example
    /// ```rust
    /// use shpr::SenojianPhaseVector;
    /// let k = SenojianPhaseVector::from_seed(1, "key", 256);
    /// let v = SenojianPhaseVector::from_seed(1, "val", 256);
    /// let bound = k.bind(&v, 0.0).unwrap();
    /// assert_eq!(bound.dim(), 256);
    /// ```
    pub fn bind(&self, rhs: &Self, phase_shift: f64) -> Result<Self, ShprError> {
        if self.dim() != rhs.dim() {
            return Err(ShprError::DimMismatch {
                expected: self.dim(),
                got: rhs.dim(),
            });
        }
        let mut phases = vec![0.0; self.dim()];
        for i in 0..self.dim() {
            phases[i] = normalize_angle(self.phases[i] + rhs.phases[i] + phase_shift);
        }
        Ok(Self { phases })
    }

    /// Performs continuous phase unbinding ($\theta_{\text{bound}} - \theta_{\text{key}} - \Delta\phi \pmod{2\pi}$).
    ///
    /// # Errors
    /// Returns [`ShprError::DimMismatch`] if `self.dim() != key.dim()`.
    ///
    /// # Example
    /// ```rust
    /// use shpr::SenojianPhaseVector;
    /// let k = SenojianPhaseVector::from_seed(1, "key", 256);
    /// let v = SenojianPhaseVector::from_seed(1, "val", 256);
    /// let bound = k.bind(&v, 0.0).unwrap();
    /// let unbound = bound.unbind(&k, 0.0).unwrap();
    /// assert!((unbound.resonance(&v) - 1.0).abs() < 1e-9);
    /// ```
    pub fn unbind(&self, key: &Self, phase_shift: f64) -> Result<Self, ShprError> {
        if self.dim() != key.dim() {
            return Err(ShprError::DimMismatch {
                expected: self.dim(),
                got: key.dim(),
            });
        }
        let mut phases = vec![0.0; self.dim()];
        for i in 0..self.dim() {
            phases[i] = normalize_angle(self.phases[i] - key.phases[i] - phase_shift);
        }
        Ok(Self { phases })
    }

    /// Computes the mean continuous phase resonance score $\frac{1}{D} \sum_{i=1}^D \cos(\theta_{1,i} - \theta_{2,i})$.
    ///
    /// Returns `1.0` for identical phase vectors and near `0.0` for orthogonal phase vectors.
    ///
    /// # Example
    /// ```rust
    /// use shpr::SenojianPhaseVector;
    /// let v1 = SenojianPhaseVector::from_seed(1, "a", 512);
    /// let v2 = SenojianPhaseVector::from_seed(1, "b", 512);
    /// assert!((v1.resonance(&v1) - 1.0).abs() < 1e-9);
    /// assert!(v1.resonance(&v2).abs() < 0.2);
    /// ```
    #[inline]
    pub fn resonance(&self, other: &Self) -> f64 {
        let dim = self.dim();
        if dim == 0 || dim != other.dim() {
            return 0.0;
        }

        let p1 = &self.phases;
        let p2 = &other.phases;
        let mut sum = 0.0;

        let chunks = dim / 4;
        let remainder = dim % 4;

        for c in 0..chunks {
            let idx = c * 4;
            sum += (p1[idx] - p2[idx]).cos()
                + (p1[idx + 1] - p2[idx + 1]).cos()
                + (p1[idx + 2] - p2[idx + 2]).cos()
                + (p1[idx + 3] - p2[idx + 3]).cos();
        }

        for i in (dim - remainder)..dim {
            sum += (p1[i] - p2[i]).cos();
        }

        sum / (dim as f64)
    }

    /// Wraps any phase angle in radians to the principal interval $[-\pi, \pi]$.
    #[inline]
    pub fn normalize_angle(angle: f64) -> f64 {
        crate::phase_math::normalize_angle(angle)
    }
}

/// Constant Memory $O(1)$ SHPR State Accumulator.
///
/// Accumulates streaming key-value phase associations into complex phasor registers $(\mathbf{r}, \mathbf{i})$:
/// $$\mathbf{r}_i = \sum \cos(\theta_{k,i} + \theta_{v,i}), \quad \mathbf{i}_i = \sum \sin(\theta_{k,i} + \theta_{v,i})$$
#[derive(Clone, Debug)]
pub struct SHPRGraphAttention {
    /// Vector dimension $D$.
    pub dim: usize,
    /// Accumulator for real parts $\cos(\theta)$.
    pub real_state: Vec<f64>,
    /// Accumulator for imaginary parts $\sin(\theta)$.
    pub imag_state: Vec<f64>,
}

impl SHPRGraphAttention {
    /// Creates a new, empty graph attention state accumulator for vectors of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            real_state: vec![0.0; dim],
            imag_state: vec![0.0; dim],
        }
    }

    /// Accumulates a key-value phase association with an optional exponential decay factor.
    ///
    /// # Errors
    /// Returns [`ShprError::DimMismatch`] if `key.dim() != self.dim` or `value.dim() != self.dim`.
    #[inline]
    pub fn accumulate(
        &mut self,
        key: &SenojianPhaseVector,
        value: &SenojianPhaseVector,
        decay: f64,
    ) -> Result<(), ShprError> {
        let dim = self.dim;
        if key.dim() != dim || value.dim() != dim {
            return Err(ShprError::DimMismatch {
                expected: dim,
                got: key.dim(),
            });
        }

        let k_phases = &key.phases;
        let v_phases = &value.phases;
        let r_state = &mut self.real_state;
        let i_state = &mut self.imag_state;

        let chunks = dim / 4;
        let remainder = dim % 4;

        for c in 0..chunks {
            let idx = c * 4;
            for offset in 0..4 {
                let i = idx + offset;
                let angle = SenojianPhaseVector::normalize_angle(k_phases[i] + v_phases[i]);
                r_state[i] = r_state[i] * decay + angle.cos();
                i_state[i] = i_state[i] * decay + angle.sin();
            }
        }

        for i in (dim - remainder)..dim {
            let angle = SenojianPhaseVector::normalize_angle(k_phases[i] + v_phases[i]);
            r_state[i] = r_state[i] * decay + angle.cos();
            i_state[i] = i_state[i] * decay + angle.sin();
        }

        Ok(())
    }

    /// Fast phasor dot product scoring against a query key without full unbinding.
    #[inline]
    pub fn fast_phasor_dot_product(&self, key: &SenojianPhaseVector) -> f64 {
        let k_phases = &key.phases;
        let r_state = &self.real_state;
        let i_state = &self.imag_state;
        let dim = self.dim;

        let mut sum = 0.0;
        let chunks = dim / 4;
        let remainder = dim % 4;

        for c in 0..chunks {
            let idx = c * 4;
            for offset in 0..4 {
                let i = idx + offset;
                let k_angle = k_phases[i];
                sum += k_angle.cos() * r_state[i] + k_angle.sin() * i_state[i];
            }
        }

        for i in (dim - remainder)..dim {
            let k_angle = k_phases[i];
            sum += k_angle.cos() * r_state[i] + k_angle.sin() * i_state[i];
        }

        sum / (dim as f64)
    }

    /// Queries the accumulated memory state using `key` to retrieve the associated value vector.
    ///
    /// # Errors
    /// Returns [`ShprError::DimMismatch`] if `key.dim() != self.dim`.
    pub fn query(&self, key: &SenojianPhaseVector) -> Result<SenojianPhaseVector, ShprError> {
        let mut mem_phases = vec![0.0; self.dim];
        for i in 0..self.dim {
            mem_phases[i] = normalize_angle(self.imag_state[i].atan2(self.real_state[i]));
        }
        let mem_vec = SenojianPhaseVector { phases: mem_phases };
        mem_vec.unbind(key, 0.0)
    }

    /// Queries memory using `key` and ranks candidate value vectors by phase resonance score.
    ///
    /// Returns a vector of `(candidate_index, resonance_score)` pairs sorted descending by score.
    ///
    /// # Errors
    /// Returns [`ShprError::DimMismatch`] if dimensions mismatch.
    pub fn rank_candidates(
        &self,
        key: &SenojianPhaseVector,
        candidates: &[SenojianPhaseVector],
    ) -> Result<Vec<(usize, f64)>, ShprError> {
        let retrieved = self.query(key)?;
        let mut scores: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, cand)| (idx, retrieved.resonance(cand)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores)
    }
}

/// Segmented Hierarchical Centroid-Routed Memory Bank.
///
/// Divides long streaming context sequences into fixed-size chunks, maintaining a key centroid for sub-linear
/// hierarchical retrieval over massive token horizons.
#[derive(Clone, Debug)]
pub struct HierarchicalPhaseMemoryBank {
    /// Dimension $D$ of vectors.
    pub dim: usize,
    /// Maximum capacity of each memory segment chunk.
    pub chunk_capacity: usize,
    /// Memory chunks containing graph attention state accumulators.
    pub chunks: Vec<SHPRGraphAttention>,
    /// Centroid summaries for fast routing.
    pub chunk_key_centroids: Vec<SHPRGraphAttention>,
    /// Number of elements stored in the current active chunk.
    pub items_in_current_chunk: usize,
}

impl HierarchicalPhaseMemoryBank {
    /// Creates a new hierarchical memory bank with a given vector dimension and segment capacity.
    ///
    /// # Example
    /// ```rust
    /// use shpr::HierarchicalPhaseMemoryBank;
    /// let memory = HierarchicalPhaseMemoryBank::new(512, 64);
    /// assert_eq!(memory.chunk_capacity, 64);
    /// ```
    pub fn new(dim: usize, chunk_capacity: usize) -> Self {
        Self {
            dim,
            chunk_capacity,
            chunks: vec![SHPRGraphAttention::new(dim)],
            chunk_key_centroids: vec![SHPRGraphAttention::new(dim)],
            items_in_current_chunk: 0,
        }
    }

    /// Ingests a key-value phase association into the active memory segment.
    ///
    /// Automatically allocates a new segment when `chunk_capacity` is reached.
    pub fn accumulate(
        &mut self,
        key: &SenojianPhaseVector,
        value: &SenojianPhaseVector,
        decay: f64,
    ) -> Result<(), ShprError> {
        if self.items_in_current_chunk >= self.chunk_capacity {
            self.chunks.push(SHPRGraphAttention::new(self.dim));
            self.chunk_key_centroids
                .push(SHPRGraphAttention::new(self.dim));
            self.items_in_current_chunk = 0;
        }

        let curr_idx = self.chunks.len() - 1;
        self.chunks[curr_idx].accumulate(key, value, decay)?;

        let dummy_val = SenojianPhaseVector::zeros(self.dim);
        self.chunk_key_centroids[curr_idx].accumulate(key, &dummy_val, 1.0)?;

        self.items_in_current_chunk += 1;
        Ok(())
    }

    /// Performs sub-linear hierarchical retrieval using centroid routing.
    ///
    /// Evaluates `top_k` candidate chunks by key centroid match before unbinding the target value.
    /// Returns `(best_resonance_score, winning_chunk_index)`.
    pub fn query_hierarchical(
        &self,
        key: &SenojianPhaseVector,
        target_value: &SenojianPhaseVector,
        top_k: usize,
    ) -> Result<(f64, usize), ShprError> {
        let num_chunks = self.chunks.len();
        if num_chunks == 0 {
            return Ok((-1.0, 0));
        }

        let mut chunk_scores: Vec<(usize, f64)> = Vec::with_capacity(num_chunks);
        for (c_idx, centroid) in self.chunk_key_centroids.iter().enumerate() {
            let score = centroid.fast_phasor_dot_product(key);
            chunk_scores.push((c_idx, score));
        }

        chunk_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k_eval = top_k.min(num_chunks);
        let mut max_res = -1.0;
        let mut winning_chunk = 0;

        for i in 0..k_eval {
            let chunk_idx = chunk_scores[i].0;
            let retrieved = self.chunks[chunk_idx].query(key)?;
            let res = retrieved.resonance(target_value);
            if res > max_res {
                max_res = res;
                winning_chunk = chunk_idx;
            }
        }

        Ok((max_res, winning_chunk))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shpr_crate_unbinding() {
        let dim = 1024;
        let k = SenojianPhaseVector::from_seed(42, "ast_caller", dim);
        let v = SenojianPhaseVector::from_seed(42, "ast_callee", dim);

        let bound = k.bind(&v, 0.0).unwrap();
        let unbound = bound.unbind(&k, 0.0).unwrap();

        let res = unbound.resonance(&v);
        assert!((res - 1.0).abs() < 1e-9);
    }
}
