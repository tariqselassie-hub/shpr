use shpr::{
    AVX2PhaseVector, HierarchicalPhaseMemoryBank, SenojianPhaseVector,
    SHPRGraphAttention, ScalarPhaseVector,
};
#[cfg(target_arch = "aarch64")]
use shpr::neon::NEONPhaseVector;

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

const DIM: usize = 1024;
const CHUNK_CAP: usize = 64;
const ITERATIONS: usize = 20_000;
const HIER_TOKENS: usize = 200;

fn format_duration(d: Duration) -> String {
    if d.as_secs_f64() < 1.0 {
        format!("{:.2} µs", d.as_secs_f64() * 1_000_000.0)
    } else {
        format!("{:.4} s", d.as_secs_f64())
    }
}

fn fmt_rate(ops: usize, d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs == 0.0 {
        return "N/A".into();
    }
    let mops = (ops as f64) / secs / 1_000_000.0;
    format!("{:.2} Mops/s", mops)
}

fn bench<T, F>(name: &str, mut f: F)
where
    F: FnMut() -> T,
{
    let mut times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _ = black_box(f());
        times.push(start.elapsed());
    }
    times.sort_unstable();
    let total: Duration = times.iter().sum();
    let p50 = times[ITERATIONS / 2];
    let p95 = times[(ITERATIONS * 95) / 100];
    let p99 = times[(ITERATIONS * 99) / 100];
    let min = times[0];
    let max = times[times.len() - 1];
    let avg = total / ITERATIONS as u32;

    println!("    {}:", name);
    println!(
        "      p50={} p95={} p99={} min={} max={} avg={} total={}",
        format_duration(p50),
        format_duration(p95),
        format_duration(p99),
        format_duration(min),
        format_duration(max),
        format_duration(avg),
        format_duration(total)
    );
    println!("      throughput={}", fmt_rate(ITERATIONS, total));
}

fn bench_bind_f64() {
    println!("\n[Phase Vector f64] bind/unbind/resonance");
    let k = SenojianPhaseVector::from_seed(1, "bench_key", DIM);
    let v = SenojianPhaseVector::from_seed(1, "bench_val", DIM);

    bench("SenojianPhaseVector::bind", || {
        black_box(&k).bind(black_box(&v), 0.0)
    });

    let bound = k.bind(&v, 0.0).unwrap();
    bench("SenojianPhaseVector::unbind", || {
        black_box(&bound).unbind(black_box(&k), 0.0)
    });

    bench("SenojianPhaseVector::resonance", || {
        let _ = black_box(&bound).resonance(black_box(&k));
    });
}

fn bench_acc_f64() {
    println!("\n[Accumulator f64] accumulate/query/fast_dot");
    let k = SenojianPhaseVector::from_seed(2, "acc_key", DIM);
    let v = SenojianPhaseVector::from_seed(2, "acc_val", DIM);
    let mut acc = SHPRGraphAttention::new(DIM);
    let candidates = vec![
        SenojianPhaseVector::from_seed(2, "a", DIM),
        SenojianPhaseVector::from_seed(2, "b", DIM),
        SenojianPhaseVector::from_seed(2, "c", DIM),
    ];

    for _ in 0..100 {
        acc.accumulate(&k, &v, 1.0).unwrap();
    }

    bench("SHPRGraphAttention::accumulate", || {
        let _ = black_box(&mut acc).accumulate(black_box(&k), black_box(&v), 1.0);
    });

    bench("SHPRGraphAttention::query", || {
        let _ = black_box(&acc).query(black_box(&k));
    });

    bench("SHPRGraphAttention::fast_phasor_dot_product", || {
        let _ = black_box(&acc).fast_phasor_dot_product(black_box(&k));
    });

    bench("SHPRGraphAttention::rank_candidates", || {
        let _ = black_box(&acc).rank_candidates(black_box(&k), black_box(&candidates));
    });
}

fn bench_scalar() {
    println!("\n[ScalarBackend] addition");
    let mut a = ScalarPhaseVector::new(DIM);
    let b = ScalarPhaseVector::from_seed(3, "scalar_b", DIM);
    bench("ScalarPhaseVector::add_phases_scalar", || {
        black_box(&mut a).add_phases_scalar(black_box(&b));
    });
}

fn bench_avx2() {
    println!("\n[AVX2Backend] phase addition + cosine resonance");
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        println!("    SKIP: avx2/fma not available on this host");
        return;
    }

    let mut a = AVX2PhaseVector::from_seed(4, "avx_a");
    let b = AVX2PhaseVector::from_seed(4, "avx_b");

    bench("AVX2PhaseVector::add_phases_avx2", || unsafe {
        black_box(&mut a).add_phases_avx2(black_box(&b));
    });

    bench("AVX2PhaseVector::phase_resonance_avx2", || unsafe {
        let _ = AVX2PhaseVector::phase_resonance_avx2(black_box(&a), black_box(&b));
    });
}

fn bench_neon() {
    println!("\n[NEONBackend] phase addition + cosine resonance");
    #[cfg(target_arch = "aarch64")]
    {
        let mut a = NEONPhaseVector::from_seed(5, "neon_a");
        let b = NEONPhaseVector::from_seed(5, "neon_b");

        bench("NEONPhaseVector::add_phases_neon", || unsafe {
            black_box(&mut a).add_phases_neon(black_box(&b));
        });

        bench("NEONPhaseVector::phase_resonance_neon", || unsafe {
            let _ = NEONPhaseVector::phase_resonance_neon(black_box(&a), black_box(&b));
        });
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("    SKIP: not aarch64");
    }
}

fn bench_hier() {
    println!("\n[HierarchicalMemoryBank] ingestion + query throughput");
    let mut bank = HierarchicalPhaseMemoryBank::new(DIM, CHUNK_CAP);

    let mut keys = Vec::with_capacity(HIER_TOKENS);
    let mut vals = Vec::with_capacity(HIER_TOKENS);
    for i in 0..HIER_TOKENS {
        keys.push(SenojianPhaseVector::from_seed(7, &format!("hk_{}", i), DIM));
        vals.push(SenojianPhaseVector::from_seed(7, &format!("hv_{}", i), DIM));
    }

    bench("HierarchicalPhaseMemoryBank::accumulate", || {
        for i in 0..HIER_TOKENS {
            let _ = black_box(&mut bank).accumulate(
                black_box(&keys[i]),
                black_box(&vals[i]),
                0.999,
            );
        }
    });

    bench("HierarchicalPhaseMemoryBank::query_hierarchical", || {
        let _ = black_box(&bank).query_hierarchical(
            black_box(&keys[0]),
            black_box(&vals[0]),
            2,
        );
    });
}

fn bench_seeding() {
    println!("\n[Seeding] f64 + AVX2 f32 initialization");
    bench("SenojianPhaseVector::from_seed f64", || {
        let _ = black_box(SenojianPhaseVector::from_seed(9, "seed_key", DIM));
    });

    #[cfg(target_arch = "x86_64")]
    {
        bench("AVX2PhaseVector::from_seed f32", || {
            let _ = black_box(AVX2PhaseVector::from_seed(9, "seed_key"));
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("    SKIP: AVX2 from_seed only measured on x86_64 host");
    }
}

fn print_runtime_features() {
    println!("\n[RuntimeFeatures]");
    println!(
        "  avx2={} fma={} sse2={} neon={}",
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("sse2"),
        cfg!(target_arch = "aarch64")
    );
}

fn print_metrics_summary() {
    println!("\n[MetricsSummary]");
    println!("  DIM={}", DIM);
    println!("  ITERATIONS={}", ITERATIONS);
    println!("  HIER_TOKENS={}", HIER_TOKENS);
    println!("  CHUNK_CAP={}", CHUNK_CAP);
}

fn main() {
    println!("==========================================================================");
    println!("  SHPR COMPREHENSIVE BENCHMARK");
    println!(
        "  dim={} iterations={} hierarchical_tokens={}",
        DIM, ITERATIONS, HIER_TOKENS
    );
    println!("==========================================================================");

    print_runtime_features();
    print_metrics_summary();

    bench_seeding();
    bench_bind_f64();
    bench_scalar();
    bench_avx2();
    bench_neon();
    bench_acc_f64();
    bench_hier();

    println!("\n==========================================================================");
    println!("  BENCHMARKS COMPLETE");
    println!("==========================================================================");
}
