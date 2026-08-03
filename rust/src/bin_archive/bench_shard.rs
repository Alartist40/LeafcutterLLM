//! Benchmark the ShardEngine with synthetic weights
//!
//! Usage:
//!   cargo run --release --bin bench_shard -- --layers 4 --hidden 512 --seq 128

use clap::Parser;
use leafcutter::inference::shard_engine::ShardEngine;
use leafcutter::model::loader::ModelConfig;
use leafcutter::model::tensor::Tensor;
use leafcutter::shard::format::QuantFormat;
use leafcutter::shard::loader::{LayerManifest, Manifest, SpecialManifest};
use leafcutter::shard::writer::ShardWriter;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "bench_shard")]
#[command(about = "Benchmark ShardEngine inference performance")]
struct Args {
    #[arg(long, default_value_t = 4)]
    layers: usize,

    #[arg(long, default_value_t = 512)]
    hidden: usize,

    #[arg(long, default_value_t = 1024)]
    intermediate: usize,

    #[arg(long, default_value_t = 128)]
    seq_len: usize,

    #[arg(long, default_value_t = 10)]
    tokens: usize,

    #[arg(long, default_value = "q8_0", value_parser = ["f32", "q8_0", "q4_0"])]
    quant: String,

    #[arg(long, default_value_t = 1)]
    heads: usize,

    /// Layer cache policy: fifo, lfru, or none.
    /// Also overridable via LEAFCUTTER_CACHE environment variable.
    #[arg(long, default_value = "default", value_parser = ["default", "fifo", "lfru", "none"])]
    cache: String,

    /// Number of layer slots in the cache.
    /// `0` = disabled, `1` = single-layer pin, `>=num_layers` = full cache.
    #[arg(long, default_value_t = 1)]
    cache_slots: usize,

    /// Replay pattern: how many forward passes to run, and the access
    /// pattern. Modes:
    ///   `sequential`  — pass 0..N monotonically (typical decode)
    ///   `strided`     — visit every layer each pass, in stride order
    ///   `random`      — randomly revisit layers (adversarial cache)
    #[arg(long, default_value = "sequential", value_parser = ["sequential", "strided", "random"])]
    pattern: String,
}

fn create_identity_weights(hidden: usize, intermediate: usize) -> HashMap<String, Tensor> {
    let mut w = HashMap::new();

    fn eye(rows: usize, cols: usize) -> Tensor {
        let mut data = vec![0.0f32; rows * cols];
        let n = rows.min(cols);
        for i in 0..n {
            data[i * cols + i] = 1.0;
        }
        Tensor::from_vec(data, vec![rows, cols])
    }

    fn zeros(rows: usize, cols: usize) -> Tensor {
        Tensor::from_vec(vec![0.0f32; rows * cols], vec![rows, cols])
    }

    fn ones(len: usize) -> Tensor {
        Tensor::from_vec(vec![1.0f32; len], vec![len])
    }

    w.insert("self_attn.q_proj.weight".to_string(), eye(hidden, hidden));
    w.insert("self_attn.k_proj.weight".to_string(), eye(hidden, hidden));
    w.insert("self_attn.v_proj.weight".to_string(), eye(hidden, hidden));
    w.insert("self_attn.o_proj.weight".to_string(), eye(hidden, hidden));
    w.insert("mlp.gate_proj.weight".to_string(), zeros(hidden, intermediate));
    w.insert("mlp.up_proj.weight".to_string(), zeros(hidden, intermediate));
    w.insert("mlp.down_proj.weight".to_string(), zeros(intermediate, hidden));
    w.insert("input_layernorm.weight".to_string(), ones(hidden));
    w.insert("post_attention_layernorm.weight".to_string(), ones(hidden));

    w
}

fn main() {
    let args = Args::parse();

    let quant_format = match args.quant.as_str() {
        "q8_0" => QuantFormat::Q8_0,
        "q4_0" => QuantFormat::Q4_0,
        _ => QuantFormat::F32,
    };

    println!("🔪 Leafcutter ShardEngine Benchmark");
    println!("   Layers:      {}", args.layers);
    println!("   Hidden:      {}", args.hidden);
    println!("   Intermediate: {}", args.intermediate);
    println!("   Quant:       {:?}", quant_format);
    println!("   Seq len:     {}", args.seq_len);
    println!("   Tokens:      {}", args.tokens);

    let output_dir = std::env::temp_dir().join("leafcutter_bench");
    std::fs::create_dir_all(&output_dir).unwrap();
    let output_dir = output_dir.to_str().unwrap();

    let config = ModelConfig {
        hidden_size: args.hidden,
        num_hidden_layers: args.layers,
        num_attention_heads: args.heads,
        num_key_value_heads: args.heads,
        intermediate_size: args.intermediate,
        max_seq_len: 2048,
        vocab_size: args.hidden,
        rope_theta: 10000.0,
        head_dim: args.hidden / args.heads,
        kv_head_dim: args.hidden / args.heads,
        ..Default::default()
    };

    let writer = ShardWriter::with_quant(config.clone(), output_dir, quant_format);

    println!("\n📦 Creating synthetic shards...");
    let mut layer_files = Vec::new();
    for i in 0..args.layers {
        let weights = create_identity_weights(args.hidden, args.intermediate);
        let (filename, size) = writer.write_layer_shard(i, &weights).unwrap();
        layer_files.push(LayerManifest { idx: i, file: filename, size });
    }

    let mut embed = HashMap::new();
    let mut embed_data = vec![0.0f32; args.hidden * args.hidden];
    for i in 0..args.hidden {
        embed_data[i * args.hidden] = i as f32;
    }
    embed.insert("model.embed_tokens.weight".to_string(), Tensor::from_vec(embed_data, vec![args.hidden, args.hidden]));
    let (embed_file, embed_size) = writer.write_special_shard("embed", &embed).unwrap();

    let mut norm = HashMap::new();
    norm.insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0f32; args.hidden], vec![args.hidden]));
    let (norm_file, norm_size) = writer.write_special_shard("norm", &norm).unwrap();

    let mut lm_head = HashMap::new();
    let mut lm_data = vec![0.0f32; args.hidden * args.hidden];
    for i in 0..args.hidden {
        lm_data[i * args.hidden + i] = 1.0;
    }
    lm_head.insert("lm_head.weight".to_string(), Tensor::from_vec(lm_data, vec![args.hidden, args.hidden]));
    let (lm_head_file, lm_head_size) = writer.write_special_shard("lm_head", &lm_head).unwrap();

    let manifest = Manifest {
        model: "bench".to_string(),
        num_layers: args.layers,
        hidden_size: args.hidden,
        vocab_size: args.hidden,
        num_attention_heads: args.heads,
        num_key_value_heads: args.heads,
        intermediate_size: args.intermediate,
        max_seq_len: 2048,
        rope_theta: 10000.0,
        shard_dir: output_dir.to_string(),
        layers: layer_files,
        special: SpecialManifest {
            embed: embed_file,
            embed_size,
            norm: norm_file,
            norm_size,
            lm_head: lm_head_file,
            lm_head_size,
        },
    };

    let manifest_path = std::env::temp_dir().join("leafcutter_bench").join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

    println!("✅ Shards ready");

    // Resolve the cache policy. CLI flag wins over env var.
    let policy = match args.cache.as_str() {
        "fifo" => leafcutter::shard::loader::CachePolicy::Fifo,
        "lfru" => leafcutter::shard::loader::CachePolicy::Lfru,
        "none" => leafcutter::shard::loader::CachePolicy::None,
        _ => leafcutter::shard::loader::CachePolicy::from_env(),
    };
    println!("   Cache policy: {:?}", policy);
    println!("   Cache slots:  {}", args.cache_slots);
    println!("   Replay:       {}", args.pattern);

    let mut engine = ShardEngine::load_with_cache_and_policy(
        manifest_path.to_str().unwrap(),
        policy,
        args.cache_slots,
    )
    .unwrap();

    // Warmup: pass through all layers once to seed the cache
    println!("🔥 Warming up...");
    let _ = engine.forward(&[1]);
    engine.reset_kv_cache();

    println!("\n🏁 Benchmarking {} passes...", args.tokens);
    let start = Instant::now();
    for token_idx in 0..args.tokens {
        // Each pass visits every layer (this is what the forward loop does
        // — N layers → N cache lookups). Per call we sample the layer
        // index based on the requested access pattern.
        let layer_idx = match args.pattern.as_str() {
            "sequential" => token_idx % args.layers,
            "strided" => (token_idx * (args.layers / 4).max(1)) % args.layers,
            "random" => {
                // deterministic-pseudo: avoid bringing rand into scope.
                ((token_idx * 2654435761) >> 16) as usize % args.layers
            }
            _ => token_idx % args.layers,
        };
        let _ = engine.forward(&[layer_idx]);
    }
    let elapsed = start.elapsed();

    let tok_per_sec = args.tokens as f64 / elapsed.as_secs_f64();
    let ms_per_tok = elapsed.as_millis() as f64 / args.tokens as f64;

    println!("\n📊 Results:");
    println!("   Total time:  {:?}", elapsed);
    println!("   Tok/sec:     {:.2}", tok_per_sec);
    println!("   ms/tok:      {:.1}", ms_per_tok);
    println!("   KV cache:    {} MB", engine.kv_cache_memory_mb());

    // Print cache hit/miss/eviction stats when LFRU is active
    if let Some(stats) = engine.cache_stats() {
        println!("\n🗄  Cache stats:");
        println!("   Hits:        {}", stats.hits);
        println!("   Misses:      {}", stats.misses);
        println!("   Evictions:   {}", stats.evictions);
        println!("   Hit rate:    {:.1}%", stats.hit_rate() * 100.0);
        println!("   Resident:    {}/{}", engine.cache_resident(), engine.cache_capacity());
        println!("   Clock:       {}", stats.clock);
    } else {
        println!(
            "\n🗄  Cache: {:?} policy (no public stats)",
            engine.policy()
        );
    }
}
