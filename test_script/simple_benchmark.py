#!/usr/bin/env python3
import subprocess
import re
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
MATRIX_SIZES = [128, 256, 512, 1024, 2048]
NUM_RUNS = 3

BASE_DIR = Path(__file__).parent.parent  
RUST_DIR = BASE_DIR / "rust"             
CPP_DIR = BASE_DIR / "c++"              
RUST_BINARY = RUST_DIR / "target" / "release" / "rust"
CPP_BINARY = CPP_DIR / "matrix"
RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"
PLOTS_DIR = Path(__file__).parent / "plots"

# ============================================================================
# BUILD FUNCTIONS
# ============================================================================
def build_programs():
    print("=" * 80)
    print("BUILDING PROGRAMS")
    print("=" * 80)
    
    # Build C++
    print("\nBuilding C++ program...")
    try:
        subprocess.run(
            ["g++", "-O3", "-std=c++17", "-pthread", "matrix.cpp", "-o", "matrix"],
            cwd=CPP_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ C++ build successful: {CPP_BINARY}")
    except subprocess.CalledProcessError as e:
        print(f"✗ C++ build failed:\n{e.stderr}")
        sys.exit(1)
    
    # Build Rust
    print("\nBuilding Rust program...")
    try:
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=RUST_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Rust build successful: {RUST_BINARY}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Rust build failed:\n{e.stderr}")
        sys.exit(1)

# ============================================================================
# TEST EXECUTION
# ============================================================================
def parse_output(output: str) -> Optional[Tuple[int, int, int]]:
    lines = output.strip().split('\n')
    for line in reversed(lines):
        numbers = re.findall(r'\d+', line)
        if len(numbers) == 3:
            return tuple(map(int, numbers))
    return None

def run_single_test(binary: Path, size: int) -> Optional[Tuple[int, int, int]]:
    try:
        result = subprocess.run(
            [str(binary), str(size), "test"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        return parse_output(result.stdout)
    except subprocess.TimeoutExpired:
        print(" TIMEOUT", end="")
        return None
    except Exception as e:
        print(f" ERROR: {e}", end="")
        return None

def run_benchmarks() -> Dict:
    results = {'cpp': {}, 'rust': {}}
    
    total_tests = len(MATRIX_SIZES) * 2
    current_test = 0
    
    for lang, binary in [('cpp', CPP_BINARY), ('rust', RUST_BINARY)]:
        print(f"\n{'=' * 80}")
        print(f"TESTING {lang.upper()}")
        print(f"{'=' * 80}")
        
        for size in MATRIX_SIZES:
            current_test += 1
            progress = (current_test / total_tests) * 100
            print(f"\n[{progress:.1f}%] {lang.upper():5s} | {size:4d}×{size:<4d} ", end="")
            
            all_runs = []
            for _ in range(NUM_RUNS):
                print(".", end="", flush=True)
                result = run_single_test(binary, size)
                if result:
                    all_runs.append(result)
            
            if all_runs:
                avg_standard = sum(r[0] for r in all_runs) / len(all_runs)
                avg_dc = sum(r[1] for r in all_runs) / len(all_runs)
                avg_strassen = sum(r[2] for r in all_runs) / len(all_runs)
                
                results[lang][size] = {
                    'standard': avg_standard,
                    'dc': avg_dc,
                    'strassen': avg_strassen,
                    'raw_runs': all_runs
                }
                print(f" ✓ [{avg_standard:.0f}, {avg_dc:.0f}, {avg_strassen:.0f}] ms")
            else:
                print(" ✗ FAILED")
                results[lang][size] = None
    
    return results

# ============================================================================
# DATA MANAGEMENT
# ============================================================================
def save_results(results: Dict):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {RESULTS_FILE}")

def load_results() -> Dict:
    if not RESULTS_FILE.exists():
        print(f"✗ Results file not found: {RESULTS_FILE}")
        sys.exit(1)
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_algorithm_comparison(results: Dict, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = {'standard': "#AB2E96", 'dc': "#247231", 'strassen': "#0C2859"}
    algo_names = {'standard': 'Standard', 'dc': 'Divide & Conquer', 'strassen': 'Strassen'}
    
    for ax, lang in [(ax1, 'cpp'), (ax2, 'rust')]:
        for algo_key in ['standard', 'dc', 'strassen']:
            times = []
            sizes = []
            
            for size in MATRIX_SIZES:
                if size in results[lang] and results[lang][size]:
                    times.append(results[lang][size][algo_key])
                    sizes.append(size)
            
            if times:
                ax.plot(sizes, times, marker='o', linewidth=2.5, markersize=10,
                       label=algo_names[algo_key], color=colors[algo_key])
        
        ax.set_xlabel('Dimenzija matrice (N×N)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Vreme izvršavanja (ms)', fontsize=13, fontweight='bold')
        ax.set_title(f'{lang.upper()}: Poređenje algoritama', fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 1_algorithm_comparison.png")

def plot_language_comparison(results: Dict, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    lang_colors = {'cpp': "#9C0060", 'rust': "#DFBE00"}
    algo_keys = ['standard', 'dc', 'strassen']
    algo_names = ['Standard', 'Divide & Conquer', 'Strassen']
    
    for ax, algo_key, algo_name in zip(axes, algo_keys, algo_names):
        for lang in ['cpp', 'rust']:
            times = []
            sizes = []
            
            for size in MATRIX_SIZES:
                if size in results[lang] and results[lang][size]:
                    times.append(results[lang][size][algo_key])
                    sizes.append(size)
            
            if times:
                ax.plot(sizes, times, marker='o', linewidth=2.5, markersize=10,
                       label=lang.upper(), color=lang_colors[lang])
        
        ax.set_xlabel('Dimenzija matrice (N×N)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Vreme izvršavanja (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'{algo_name}', fontsize=13, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
    
    plt.suptitle('C++ vs Rust: Poređenje po algoritmu', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '2_language_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 2_language_comparison.png")

def plot_relative_performance(results: Dict, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = {'dc': "#247231", 'strassen': "#0C2859"}
    algo_names = {'dc': 'Divide & Conquer', 'strassen': 'Strassen'}
    
    for ax, lang in [(ax1, 'cpp'), (ax2, 'rust')]:
        for algo_key in ['dc', 'strassen']:
            speedups = []
            sizes = []
            
            for size in MATRIX_SIZES:
                if size in results[lang] and results[lang][size]:
                    standard_time = results[lang][size]['standard']
                    algo_time = results[lang][size][algo_key]
                    if algo_time > 0:
                        speedup = standard_time / algo_time
                        speedups.append(speedup)
                        sizes.append(size)
            
            if speedups:
                ax.plot(sizes, speedups, marker='o', linewidth=2.5, markersize=10,
                       label=algo_names[algo_key], color=colors[algo_key])
        
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Standard (baseline)')
        ax.set_xlabel('Dimenzija matrice (N×N)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Speedup (Standard / Algoritam)', fontsize=13, fontweight='bold')
        ax.set_title(f'{lang.upper()}: Relativni performans', fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_relative_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 3_relative_performance.png")

def plot_speedup_bar_chart(results: Dict, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    size = 2048
    algos = ['dc', 'strassen']
    algo_names = ['Divide & Conquer', 'Strassen']
    
    for ax, lang in [(ax1, 'cpp'), (ax2, 'rust')]:
        speedups = []
        labels = []
        
        if size in results[lang] and results[lang][size]:
            standard_time = results[lang][size]['standard']
            
            for algo, algo_name in zip(algos, algo_names):
                algo_time = results[lang][size][algo]
                if algo_time > 0:
                    speedup = standard_time / algo_time
                    speedups.append(speedup)
                    labels.append(algo_name)
        
        if speedups:
            x = np.arange(len(labels))
            bars = ax.bar(x, speedups, color=["#247231", "#0C2859"], alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            ax.set_ylabel('Speedup vs Standard', fontsize=12, fontweight='bold')
            ax.set_title(f'{lang.upper()}: Speedup za {size}×{size}', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, label='Standard baseline')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=10)
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_speedup_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 4_speedup_bar_chart.png")

def generate_all_plots(results: Dict):
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    sns.set_style("whitegrid")
    PLOTS_DIR.mkdir(exist_ok=True)
    
    plot_algorithm_comparison(results, PLOTS_DIR)
    plot_language_comparison(results, PLOTS_DIR)
    plot_relative_performance(results, PLOTS_DIR)
    plot_speedup_bar_chart(results, PLOTS_DIR)
    
    print(f"\n✓ All plots saved to: {PLOTS_DIR}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
def print_summary(results: Dict):
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for size in MATRIX_SIZES:
        print(f"\n{'=' * 70}")
        print(f"Results for {size}×{size} matrix")
        print(f"{'=' * 70}")
        print(f"{'Lang':<8} {'Algorithm':<18} {'Time (ms)':<12} {'vs Standard':<12}")
        print("-" * 70)
        
        for lang in ['cpp', 'rust']:
            if size not in results[lang] or not results[lang][size]:
                continue
            
            standard_time = results[lang][size]['standard']
            
            for algo, algo_name in [('standard', 'Standard'), ('dc', 'DC'), ('strassen', 'Strassen')]:
                time = results[lang][size][algo]
                if algo == 'standard':
                    speedup_str = "baseline"
                else:
                    speedup = standard_time / time if time > 0 else 0
                    speedup_str = f"{speedup:.2f}x"
                
                print(f"{lang.upper():<8} {algo_name:<18} {time:<12.0f} {speedup_str:<12}")
    
    # Overall winner
    size = MATRIX_SIZES[-1]
    print("\n" + "=" * 80)
    print(f"FASTEST CONFIGURATION ({size}×{size})")
    print("=" * 80)
    
    fastest_time = float('inf')
    fastest_config = None
    
    for lang in ['cpp', 'rust']:
        if size in results[lang] and results[lang][size]:
            for algo in ['standard', 'dc', 'strassen']:
                time = results[lang][size][algo]
                if time < fastest_time:
                    fastest_time = time
                    fastest_config = (lang, algo)
    
    if fastest_config:
        lang, algo = fastest_config
        algo_names = {'standard': 'Standard', 'dc': 'Divide & Conquer', 'strassen': 'Strassen'}
        print(f"\n {lang.upper()} - {algo_names[algo]}: {fastest_time:.0f} ms\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("MATRIX MULTIPLICATION BENCHMARK: RUST VS C++")
    print("=" * 80)
    
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark matrix multiplication')
    parser.add_argument('--skip-build', action='store_true', help='Skip compilation')
    parser.add_argument('--skip-test', action='store_true', help='Skip testing (use existing results)')
    parser.add_argument('--plot-only', action='store_true', help='Only generate plots from existing results')
    args = parser.parse_args()
    
    try:
        if args.plot_only:
            print("\nLoading existing results...")
            results = load_results()
            generate_all_plots(results)
        else:
            if not args.skip_build:
                build_programs()
            
            if not args.skip_test:
                results = run_benchmarks()
                save_results(results)
            else:
                print("\nLoading existing results...")
                results = load_results()
            
            generate_all_plots(results)
            print_summary(results)
        
        print("\n" + "=" * 80)
        print("✓ BENCHMARK COMPLETE!")
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()