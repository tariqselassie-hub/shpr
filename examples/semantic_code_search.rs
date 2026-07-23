//! Practical Example: Lossless AST Symbol Search using Continuous Phase Vectors
//!
//! Run with:
//! ```bash
//! cargo run --example semantic_code_search
//! ```

use shpr::{SHPRGraphAttention, SenojianPhaseVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 [shpr] Semantic Code Symbol Unbinding Example");
    println!("--------------------------------------------------");

    let dim = 2048;

    // 1. Define symbolic Keys (AST Role) and Values (Code Identifier / Body)
    let caller_key = SenojianPhaseVector::from_seed(42, "AST_Role_Caller", dim);
    let caller_val = SenojianPhaseVector::from_seed(42, "fn_compute_attention", dim);

    let callee_key = SenojianPhaseVector::from_seed(42, "AST_Role_Callee", dim);
    let callee_val = SenojianPhaseVector::from_seed(42, "fn_avx2_phase_resonance", dim);

    // 2. Accumulate bound AST key-value pairs into O(1) SHPR Graph Attention memory
    let mut ast_attention = SHPRGraphAttention::new(dim);
    ast_attention.accumulate(&caller_key, &caller_val, 1.0)?;
    ast_attention.accumulate(&callee_key, &callee_val, 1.0)?;

    // 3. Query memory for "AST_Role_Callee"
    let retrieved = ast_attention.query(&callee_key)?;

    // 4. Measure phase resonance score against candidates
    let resonance_callee = retrieved.resonance(&callee_val);
    let resonance_caller = retrieved.resonance(&caller_val);
    let resonance_random = retrieved.resonance(&SenojianPhaseVector::from_seed(99, "random", dim));

    println!("  Target Symbol Query: 'AST_Role_Callee'");
    println!("  Retrieved Match     : 'fn_avx2_phase_resonance'");
    println!(
        "  Resonance Score (Target Match)   : {:.4}",
        resonance_callee
    );
    println!(
        "  Resonance Score (Other AST Role)  : {:.4}",
        resonance_caller
    );
    println!(
        "  Resonance Score (Random Noise)    : {:.4}",
        resonance_random
    );

    // 5. Rank candidates automatically
    let candidates = vec![caller_val.clone(), callee_val.clone()];
    let ranked = ast_attention.rank_candidates(&callee_key, &candidates)?;
    println!("\n  Ranked Candidates for 'AST_Role_Callee':");
    for (rank, (idx, score)) in ranked.iter().enumerate() {
        let name = if *idx == 1 {
            "fn_avx2_phase_resonance"
        } else {
            "fn_compute_attention"
        };
        println!("    #{}: {} (Score: {:.4})", rank + 1, name, score);
    }

    assert_eq!(
        ranked[0].0, 1,
        "Top ranked candidate should be fn_avx2_phase_resonance"
    );
    assert!(resonance_callee > 0.50);
    println!("\n✅ Successfully retrieved bound AST symbol from graph attention memory!");

    Ok(())
}
