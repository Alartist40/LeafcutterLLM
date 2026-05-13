#!/usr/bin/env python3
"""
Generate comparison graphs from test results
"""

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    """Load all test results from JSON files in results/ directory"""
    results = {}
    for filepath in glob.glob("results/test_*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
                model_name = data.get('model_name', os.path.basename(filepath))
                if model_name not in results:
                    results[model_name] = []
                results[model_name].append(data)
        except Exception as e:
            print(f"⚠️ Error loading {filepath}: {e}")
    return results

def plot_latency_comparison(results):
    """Plot average latency per request by model"""
    models = list(results.keys())
    latencies = [results[m][0].get('avg_latency_ms', 0) for m in models]
    
    if not models: return

    plt.figure(figsize=(10, 6))
    plt.bar(models, latencies, color='skyblue', edgecolor='navy')
    plt.axhline(y=200, color='red', linestyle='--', label='Robot Target (<200ms)')
    plt.ylabel('Avg Latency (ms)')
    plt.title('Leafcutter: Request Latency by Model')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=150)
    print("✅ Saved: latency_comparison.png")

def plot_ram_comparison(results):
    """Plot peak RAM vs model"""
    models = list(results.keys())
    ram_usage = [results[m][0].get('peak_ram_mb', 0) for m in models]
    
    if not models: return

    plt.figure(figsize=(10, 6))
    plt.bar(models, ram_usage, color='lightcoral', edgecolor='darkred')
    plt.ylabel('Peak RAM (MB)')
    plt.title('Leafcutter: RAM Usage by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ram_comparison.png', dpi=150)
    print("✅ Saved: ram_comparison.png")

def plot_throughput_comparison(results):
    """Plot tokens/second vs model"""
    models = list(results.keys())
    tps = [results[m][0].get('tokens_per_sec', 0) for m in models]
    
    if not models: return

    plt.figure(figsize=(10, 6))
    plt.bar(models, tps, color='lightgreen', edgecolor='darkgreen')
    plt.ylabel('Tokens / Second')
    plt.title('Leafcutter: Throughput by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('throughput_comparison.png', dpi=150)
    print("✅ Saved: throughput_comparison.png")

if __name__ == "__main__":
    print("📊 Analyzing Leafcutter test results...")
    if not os.path.exists("results"):
        print("❌ No results/ directory found.")
        exit(1)
        
    results = load_results()
    
    if not results:
        print("❌ No valid test results found in ./results/")
        exit(1)
    
    plot_latency_comparison(results)
    plot_ram_comparison(results)
    plot_throughput_comparison(results)
    
    print("\n🎉 All graphs generated!")
