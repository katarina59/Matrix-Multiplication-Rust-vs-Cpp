#!/usr/bin/env python3
import subprocess
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
MATRIX_SIZES = [128, 256, 512, 1024, 2048]
MAX_DEPTHS = [0, 1, 2, 3, 4]
NUM_RUNS = 3
ALGORITHMS = ['Standard', 'DC', 'Strassen']

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
    """Compile both Rust and C++ programs."""
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
        print(f"C++ build successful: {CPP_BINARY}")
    except subprocess.CalledProcessError as e:
        print(f"C++ build failed:\n{e.stderr}")
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
        print(f"Rust build successful: {RUST_BINARY}")
    except subprocess.CalledProcessError as e:
        print(f"Rust build failed:\n{e.stderr}")
        sys.exit(1)

# ============================================================================
# TEST EXECUTION
# ============================================================================
def parse_output(output: str) -> Optional[Tuple[int, int, int]]:
    """Parse the output to extract three timing values (standard, dc, strassen)."""
    lines = output.strip().split('\n')
    for line in reversed(lines):
        numbers = re.findall(r'\d+', line)
        if len(numbers) == 3:
            return tuple(map(int, numbers))
    return None

def run_single_test(binary: Path, size: int, depth: int) -> Optional[Tuple[int, int, int]]:
    """Run a single test and return the three timing values."""
    try:
        result = subprocess.run(
            [str(binary), str(size), str(depth), "test"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        return parse_output(result.stdout)
    except subprocess.TimeoutExpired:
        print(f" Timeout")
        return None
    except Exception as e:
        print(f" Error: {e}")
        return None

def run_benchmarks() -> Dict:
    results = {'cpp': {}, 'rust': {}}
    
    total_tests = len(MATRIX_SIZES) * len(MAX_DEPTHS) * 2
    current_test = 0
    
    for lang, binary in [('cpp', CPP_BINARY), ('rust', RUST_BINARY)]:
        print(f"\n{'=' * 80}")
        print(f"TESTING {lang.upper()}")
        print(f"{'=' * 80}")
        
        for size in MATRIX_SIZES:
            results[lang][size] = {}
            
            for depth in MAX_DEPTHS:
                current_test += 1
                progress = (current_test / total_tests) * 100
                print(f"\n[{progress:.1f}%] {lang.upper():4s} | Size: {size:4d} | Depth: {depth} ", end="")
                
                all_runs = []
                for run in range(NUM_RUNS):
                    print(".", end="", flush=True)
                    result = run_single_test(binary, size, depth)
                    if result:
                        all_runs.append(result)
                
                if all_runs:
                    avg_standard = sum(r[0] for r in all_runs) / len(all_runs)
                    avg_dc = sum(r[1] for r in all_runs) / len(all_runs)
                    avg_strassen = sum(r[2] for r in all_runs) / len(all_runs)
                    
                    results[lang][size][depth] = {
                        'standard': avg_standard,
                        'dc': avg_dc,
                        'strassen': avg_strassen,
                        'raw_runs': all_runs
                    }
                    print(f" [{avg_standard:.0f}, {avg_dc:.0f}, {avg_strassen:.0f}] ms")
                else:
                    print(" ")
                    results[lang][size][depth] = None
    
    return results

# ============================================================================
# DATA MANAGEMENT
# ============================================================================
def save_results(results: Dict):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")

def load_results() -> Dict:
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
            best_times = []
            sizes = []
            
            for size in MATRIX_SIZES:
                if size not in results[lang]:
                    continue
                
                min_time = float('inf')
                for depth in MAX_DEPTHS:
                    if depth in results[lang][size]:
                        data = results[lang][size][depth]
                        if data and data[algo_key] < min_time:
                            min_time = data[algo_key]
                
                if min_time != float('inf'):
                    best_times.append(min_time)
                    sizes.append(size)
            
            if best_times:
                ax.plot(sizes, best_times, marker='o', linewidth=2.5, markersize=10,
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
            best_times = []
            sizes = []
            
            for size in MATRIX_SIZES:
                if size not in results[lang]:
                    continue
                
                min_time = float('inf')
                for depth in MAX_DEPTHS:
                    if depth in results[lang][size]:
                        data = results[lang][size][depth]
                        if data and data[algo_key] < min_time:
                            min_time = data[algo_key]
                
                if min_time != float('inf'):
                    best_times.append(min_time)
                    sizes.append(size)
            
            if best_times:
                ax.plot(sizes, best_times, marker='o', linewidth=2.5, markersize=10,
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

def plot_parallelization_impact(results: Dict, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    depth_colors = ['#A23B72', "#049CB6", "#114D05", "#EDF423", "#95363B"]
    
    configs = [
        (axes[0, 0], 'cpp', 'dc', 'C++: Divide & Conquer'),
        (axes[0, 1], 'cpp', 'strassen', 'C++: Strassen'),
        (axes[1, 0], 'rust', 'dc', 'Rust: Divide & Conquer'),
        (axes[1, 1], 'rust', 'strassen', 'Rust: Strassen'),
    ]
    
    for ax, lang, algo, title in configs:
        for depth_idx, depth in enumerate(MAX_DEPTHS):
            times = []
            sizes = []
            
            for size in MATRIX_SIZES:
                if size in results[lang] and depth in results[lang][size]:
                    data = results[lang][size][depth]
                    if data:
                        times.append(data[algo])
                        sizes.append(size)
            
            if times:
                ax.plot(sizes, times, marker='o', linewidth=2, markersize=8,
                       label=f'depth={depth}', color=depth_colors[depth_idx])
        
        ax.set_xlabel('Dimenzija matrice (N×N)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Vreme izvršavanja (ms)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    
    plt.suptitle('Uticaj dubine paralelizacije', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '3_parallelization_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 3_parallelization_impact.png")

def plot_speedup_analysis(results: Dict, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    size = 2048
    algos = ['dc', 'strassen']
    algo_names = ['Divide & Conquer', 'Strassen']
    
    for ax, lang in [(ax1, 'cpp'), (ax2, 'rust')]:
        speedups = []
        labels = []
        
        for algo, algo_name in zip(algos, algo_names):
            if size not in results[lang]:
                continue
            
            # Sequential time (depth=0)
            seq_time = results[lang][size][0][algo] if 0 in results[lang][size] else None
            
            # Best parallel time
            best_time = float('inf')
            for depth in [1, 2, 3, 4]:
                if depth in results[lang][size]:
                    data = results[lang][size][depth]
                    if data and data[algo] < best_time:
                        best_time = data[algo]
            
            if seq_time and best_time != float('inf'):
                speedup = seq_time / best_time
                speedups.append(speedup)
                labels.append(algo_name)
        
        x = np.arange(len(labels))
        bars = ax.bar(x, speedups, color=["#C52CAB", "#1BAAB4"], alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Speedup (sekvencijalno / paralelno)', fontsize=12, fontweight='bold')
        ax.set_title(f'{lang.upper()}: Speedup za 2048×2048', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Baseline')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 4_speedup_analysis.png")


def generate_all_plots(results: Dict):
    """Generate all visualization plots."""
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    sns.set_style("whitegrid")
    PLOTS_DIR.mkdir(exist_ok=True)
    
    plot_algorithm_comparison(results, PLOTS_DIR)
    plot_language_comparison(results, PLOTS_DIR)
    plot_parallelization_impact(results, PLOTS_DIR)
    plot_speedup_analysis(results, PLOTS_DIR)
    
    print(f"\nAll plots saved to: {PLOTS_DIR}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
def print_summary(results: Dict):
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Table for 2048x2048 matrix
    size = 2048
    print(f"\nResults for {size}×{size} matrix:")
    print("-" * 80)
    print(f"{'Language':<10} {'Algorithm':<18} {'Best Time':<12} {'Depth':<8} {'Speedup'}")
    print("-" * 80)
    
    for lang in ['cpp', 'rust']:
        if size not in results[lang]:
            continue
        
        for algo, algo_name in [('standard', 'Standard'), ('dc', 'DC'), ('strassen', 'Strassen')]:
            best_time = float('inf')
            best_depth = None
            seq_time = None
            
            if 0 in results[lang][size]:
                seq_time = results[lang][size][0][algo]
            
            for depth in MAX_DEPTHS:
                if depth in results[lang][size]:
                    time = results[lang][size][depth][algo]
                    if time and time < best_time:
                        best_time = time
                        best_depth = depth
            
            speedup = seq_time / best_time if seq_time and best_time != float('inf') else 1.0
            
            print(f"{lang.upper():<10} {algo_name:<18} {best_time:<12.0f} {best_depth:<8} {speedup:.2f}x")
    
    # Overall winner
    print("\n" + "=" * 80)
    print("OVERALL WINNER (2048×2048)")
    print("=" * 80)
    
    fastest_time = float('inf')
    fastest_config = None
    
    for lang in ['cpp', 'rust']:
        for algo in ['standard', 'dc', 'strassen']:
            for depth in MAX_DEPTHS:
                if size in results[lang] and depth in results[lang][size]:
                    data = results[lang][size][depth]
                    if data and data[algo] < fastest_time:
                        fastest_time = data[algo]
                        fastest_config = (lang, algo, depth)
    
    if fastest_config:
        lang, algo, depth = fastest_config
        algo_names = {'standard': 'Standard', 'dc': 'Divide & Conquer', 'strassen': 'Strassen'}
        print(f"\n{lang.upper()} - {algo_names[algo]} (depth={depth}): {fastest_time:.0f} ms\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("MATRIX MULTIPLICATION BENCHMARK: RUST VS C++")
    print("=" * 80)
    
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark matrix multiplication')
    parser.add_argument('--skip-build', action='store_true', help='Skip building')
    parser.add_argument('--skip-test', action='store_true', help='Skip testing')
    parser.add_argument('--plot-only', action='store_true', help='Only generate plots')
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
        print("BENCHMARK COMPLETE!")
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n Error: {e}")
        raise

if __name__ == "__main__":
    main()