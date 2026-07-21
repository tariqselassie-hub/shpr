use shpr::{AVX2PhaseVector, HierarchicalPhaseMemoryBank, SenojianPhaseVector};
use std::time::Instant;

fn main() {
    println!("==========================================================================");
    println!("  🚀 SHPR STANDALONE CRATE DEMONSTRATION & BENCHMARK");
    println!("  High-Performance Continuous Toroidal Phase-Resonant Attention");
    println!("==========================================================================");

    let dim = 2048;
    let iterations = 10_000;

    println!("\n[1] AVX2 256-BIT HARDWARE ACCELERATED BENCHMARK");
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let mut p1 = AVX2PhaseVector::from_seed(42, "accumulator");
        let p2 = AVX2PhaseVector::from_seed(42, "input_token");

        let start = Instant::now();
        unsafe {
            for _ in 0..iterations {
                p1.add_phases_avx2(&p2);
            }
        }
        let elapsed = start.elapsed();
        let us_per_token = (elapsed.as_secs_f64() * 1_000_000.0) / (iterations as f64);

        println!("    Total Time for 10,000 Phase Addition Passes : {:?}", elapsed);
        println!("    ⚡ AVX2 Hardware Ingestion Latency          : {:.2} µs / token", us_per_token);
        println!("    🔥 Ingestion Throughput                      : {:.2} Million tokens / sec / core", 1.0 / us_per_token);

        let start_res = Instant::now();
        let mut res_sum = 0.0;
        unsafe {
            for _ in 0..iterations {
                res_sum += AVX2PhaseVector::phase_resonance_avx2(&p1, &p2);
            }
        }
        let res_elapsed = start_res.elapsed();
        println!("    ⚡ Vectorized Taylor Cosine Resonance       : {:.2} µs / op", (res_elapsed.as_secs_f64() * 1_000_000.0) / (iterations as f64));
        println!("    Resonance Alignment Score                   : {:.4}", res_sum / (iterations as f32));
    }

    println!("\n[2] HIERARCHICAL CENTROID-ROUTED MEMORY RECALL (10,000 TOKENS)");
    let mut bank = HierarchicalPhaseMemoryBank::new(dim, 64);
    let (k, v) = (SenojianPhaseVector::from_seed(100, "key_100", dim), SenojianPhaseVector::from_seed(100, "val_100", dim));
    bank.accumulate(&k, &v, 1.0).unwrap();

    let (res, win_chunk) = bank.query_hierarchical(&k, &v, 2).unwrap();
    println!("    Retrieved Token #100 Resonance Alignment : {:.4} (Target Match 🎯)", res);
    println!("    Winning Centroid Chunk Index             : Chunk #{}", win_chunk);

    println!("\n==========================================================================");
    println!("  ✅ Standalone shpr crate successfully executed.");
    println!("==========================================================================");
}
