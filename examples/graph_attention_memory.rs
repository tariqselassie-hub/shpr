//! Practical Example: Streaming Token Context Ingestion and Hierarchical Centroid Retrieval
//!
//! Run with:
//! ```bash
//! cargo run --example graph_attention_memory
//! ```

use shpr::{HierarchicalPhaseMemoryBank, SenojianPhaseVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ [shpr] Streaming Graph Attention & Hierarchical Memory");
    println!("----------------------------------------------------------");

    let dim = 1024;
    let chunk_capacity = 32;

    // Create a hierarchical memory bank
    let mut memory = HierarchicalPhaseMemoryBank::new(dim, chunk_capacity);

    let num_tokens = 128;
    println!(
        "📥 Ingesting {} streaming key-value token associations...",
        num_tokens
    );

    let target_token_idx = 75;
    let mut target_key = SenojianPhaseVector::zeros(dim);
    let mut target_val = SenojianPhaseVector::zeros(dim);

    for i in 0..num_tokens {
        let k_str = format!("token_key_{}", i);
        let v_str = format!("token_val_{}", i);

        let k = SenojianPhaseVector::from_seed(42, &k_str, dim);
        let v = SenojianPhaseVector::from_seed(42, &v_str, dim);

        if i == target_token_idx {
            target_key = k.clone();
            target_val = v.clone();
        }

        memory.accumulate(&k, &v, 0.999)?;
    }

    println!("  Total Chunks Created: {}", memory.chunks.len());
    println!("  Target Key: 'token_key_{}'", target_token_idx);

    // Perform sub-linear hierarchical retrieval using centroid routing
    let (score, winning_chunk) = memory.query_hierarchical(&target_key, &target_val, 2)?;

    println!("  Winning Segment Chunk Index  : {}", winning_chunk);
    println!("  Hierarchical Resonance Score : {:.4}", score);

    assert_eq!(
        winning_chunk, 2,
        "Centroid routing must accurately identify target chunk #2"
    );
    assert!(
        score > 0.10,
        "Target score must stand out significantly above orthogonal noise"
    );
    println!("\n✅ Successfully routed to target segment chunk and retrieved token value!");

    Ok(())
}
