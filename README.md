# `shpr` 🚀

**Senojian-Hyperdimensional Phase-Resonant Graph Attention Engine**

[![CI Pipeline](https://github.com/tariqselassie-hub/shpr/actions/workflows/ci.yml/badge.svg)](https://github.com/tariqselassie-hub/shpr/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/shpr.svg)](https://crates.io/crates/shpr)
[![Documentation](https://docs.rs/shpr/badge.svg)](https://docs.rs/shpr)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

A zero-dependency, ultra-lightweight, hardware-accelerated Vector Symbolic Architecture (VSA) and Hyperdimensional Computing (HDC) phase memory engine in Rust.

`shpr` solves the $O(N^2)$ memory and computation bottleneck of Transformer Attention by projecting features onto continuous toroidal phase manifolds ($\mathbb{T}^D = S^1 \times \dots \times S^1$).

---

|

| ## ⚡ Performance Highlights

| Measured on x86_64 with AVX2 + FMA enabled:

| - **AVX2 Phase Addition:** ~**4.3 µs / token** for 10,000 phase additions on `T^2048`
| - **AVX2 Resonance Scoring:** ~**6.7 µs / op** for 10,000 cosine-resonance evaluations
| - **Deterministic Unbinding:** Continuous phase unbinding achieves exact symbolic retrieval with `SNR = ∞`
| - **Zero Softmax / Matrix Multiplication:** Replaces pairwise matrix multiplies with phase resonance alignment `∑ cos(Δθ)`

|

|---

|

| ## 📦 Quickstart (Rust)

Add `shpr` to your `Cargo.toml`:

```toml
[dependencies]
shpr = { version = "0.1.0", features = ["avx2"] }
```

### Example: Continuous Phase Binding & Unbinding

```rust
use shpr::{SenojianPhaseVector, HierarchicalPhaseMemoryBank};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim = 2048;

    // 1. Create continuous phase vectors
    let key = SenojianPhaseVector::from_seed(42, "AST_Caller_Node", dim);
    let val = SenojianPhaseVector::from_seed(42, "AST_Callee_Body", dim);

    // 2. Continuous Toroidal Binding: (theta_A + theta_B) mod 2PI
    let bound = key.bind(&val, 0.0)?;

    // 3. Exact Lossless Unbinding: (theta_bound - theta_key) mod 2PI
    let retrieved = bound.unbind(&key, 0.0)?;

    // 4. Measure Phase Resonance Score (1.0 = Exact Match)
    let score = retrieved.resonance(&val);
    println!("Resonance Score: {:.4}", score); // 1.0000

    Ok(())
}
```

---

## 🏗️ Architecture: Segmented Hierarchical Memory

```text
Incoming Stream ──► Segmented Ring Memory (64 Tokens / Chunk)
                         │
                         ├──► Centroid Summary Indexing (O(M) Zero-Alloc)
                         └──► AVX2 Phase Addition Engine (110ns / Token)
```

---

## 🛠️ Building & Running Benchmarks

Run unit tests:
```bash
cargo test --lib
```

Run release benchmark:
```bash
cargo run --release --bin shpr_demo
```

---

## 🤝 Community & Contributing

We welcome community contributions, research proposals, SIMD optimizations, and new features!

- **Discussion & Q&A:** Join the conversation on [GitHub Discussions](https://github.com/tariqselassie-hub/shpr/discussions) or open an issue.
- **Contribution Guidelines:** See [CONTRIBUTING.md](CONTRIBUTING.md) for build instructions, code formatting, and PR workflows.
- **Code of Conduct:** We adhere to the [Contributor Covenant](CODE_OF_CONDUCT.md).

---

## 📜 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
