# Contributing to `shpr` 🚀

Thank you for your interest in contributing to **`shpr`** (Senojian-Hyperdimensional Phase-Resonant Graph Attention Engine)! 

We welcome contributions from developers, researchers, and open-source enthusiasts of all skill levels. Whether you are fixing bugs, improving SIMD performance, writing docs, or exploring new Hyperdimensional Computing (HDC) and Vector Symbolic Architecture (VSA) algorithms, your help is appreciated!

---

## 🛠️ How to Get Started

### 1. Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain)
- Cargo (comes with Rust)

### 2. Fork and Clone
```bash
git clone https://github.com/YOUR-USERNAME/shpr.git
cd shpr
```

### 3. Build & Run Tests
Ensure all existing unit tests pass before making changes:
```bash
# Run lib tests with AVX2 feature (x86_64)
cargo test --lib --features avx2

# Run lib tests with NEON feature (ARM64)
cargo test --lib --features neon

# Run demo benchmark
cargo run --release --bin shpr_demo
```

---

## 💡 Areas to Contribute

- ⚡ **Hardware Acceleration:** Further SIMD optimizations (AVX-512, NEON, WebAssembly SIMD128, Metal/CUDA bindings).
- 🐍 **Python Bindings:** Exposing `shpr` to Python via `pyo3` and `maturin` for PyTorch/JAX interoperability.
- 📚 **Documentation & Examples:** Writing real-world examples (RAG, text search, AST analysis, graph attention).
- 🔬 **Research & Benchmarks:** Comparing `shpr` continuous phase unbinding against standard Transformer KV caches and classical VSA models.

---

## 📋 Pull Request Process

1. **Create a Feature Branch:** `git checkout -b feature/my-cool-optimization`
2. **Follow Rust Conventions:** Ensure code is formatted with `cargo fmt` and passes `cargo clippy`.
3. **Add Tests:** Include unit or integration tests for new features or bug fixes.
4. **Submit PR:** Open a Pull Request on GitHub with a detailed description of your changes.

---

## 🤝 Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
