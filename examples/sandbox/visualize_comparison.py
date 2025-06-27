#!/usr/bin/env python3
"""
Visualization of Old vs New Implementation Comparison
===================================================

This script creates visualizations comparing the performance and accuracy
of the old standalone implementation vs the new SDK-based implementation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_comparison_data():
    """Load comparison results from JSON file."""
    results_file = Path("comparison_results/comparison_results.json")
    
    if not results_file.exists():
        print("❌ No comparison results found!")
        print("   Run compare_old_vs_new.py first to generate data.")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data

def plot_fitness_curves(data):
    """Plot fitness evolution curves for both implementations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Best fitness curves
    ax1.set_title("Best Fitness Evolution", fontsize=14, fontweight='bold')
    
    for name, result in data.items():
        if result['best_fitness_curve']:
            generations = range(1, len(result['best_fitness_curve']) + 1)
            ax1.plot(generations, result['best_fitness_curve'], 
                    label=f"{result['name']}", linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mean fitness curves
    ax2.set_title("Mean Fitness Evolution", fontsize=14, fontweight='bold')
    
    for name, result in data.items():
        if result['mean_fitness_curve']:
            generations = range(1, len(result['mean_fitness_curve']) + 1)
            ax2.plot(generations, result['mean_fitness_curve'], 
                    label=f"{result['name']}", linewidth=2, marker='s', markersize=4)
    
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Fitness")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comparison_results/fitness_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_comparison(data):
    """Plot performance comparison bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [result['name'] for result in data.values()]
    times = [result['time_per_generation'] for result in data.values()]
    accuracies = [result['final_accuracy'] for result in data.values()]
    
    # Time per generation
    bars1 = ax1.bar(names, times, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax1.set_title("Time per Generation", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Time (seconds)")
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # Final accuracy
    bars2 = ax2.bar(names, accuracies, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax2.set_title("Final Accuracy", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("comparison_results/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence_analysis(data):
    """Plot convergence analysis."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create convergence plot
    ax.set_title("Convergence Analysis", fontsize=14, fontweight='bold')
    
    for name, result in data.items():
        if result['best_fitness_curve']:
            generations = range(1, len(result['best_fitness_curve']) + 1)
            fitness_curve = result['best_fitness_curve']
            
            # Plot fitness curve
            ax.plot(generations, fitness_curve, label=f"{result['name']}", 
                   linewidth=2, marker='o', markersize=3)
            
            # Mark convergence point if exists
            if result['convergence_generation'] is not None:
                conv_gen = result['convergence_generation']
                conv_fitness = fitness_curve[conv_gen]
                ax.scatter([conv_gen + 1], [conv_fitness], s=100, 
                          marker='*', label=f"{result['name']} Convergence")
    
    # Add convergence threshold line
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, 
               label='95% Accuracy Threshold')
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comparison_results/convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(data):
    """Create a text summary report."""
    report_path = Path("comparison_results/summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EM-4/3 Doubling: Old vs New Implementation Comparison\n")
        f.write("=" * 60 + "\n\n")
        
        # Performance summary
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        
        old_result = data['old']
        new_result = data['new']
        
        f.write(f"Old Implementation:\n")
        f.write(f"  - Time per generation: {old_result['time_per_generation']:.3f}s\n")
        f.write(f"  - Final accuracy: {old_result['final_accuracy']:.4f}\n")
        f.write(f"  - Total time: {old_result['total_time']:.2f}s\n\n")
        
        f.write(f"New Implementation:\n")
        f.write(f"  - Time per generation: {new_result['time_per_generation']:.3f}s\n")
        f.write(f"  - Final accuracy: {new_result['final_accuracy']:.4f}\n")
        f.write(f"  - Total time: {new_result['total_time']:.2f}s\n\n")
        
        # Speed comparison
        speed_ratio = old_result['time_per_generation'] / new_result['time_per_generation']
        f.write("SPEED COMPARISON:\n")
        f.write("-" * 17 + "\n")
        if speed_ratio > 1.1:
            f.write(f"✅ OLD is {speed_ratio:.1f}x FASTER\n")
        elif speed_ratio < 0.9:
            f.write(f"❌ OLD is {1/speed_ratio:.1f}x SLOWER\n")
        else:
            f.write(f"≈ Similar performance ({speed_ratio:.1f}x)\n")
        
        # Accuracy comparison
        f.write("\nACCURACY COMPARISON:\n")
        f.write("-" * 20 + "\n")
        acc_diff = new_result['final_accuracy'] - old_result['final_accuracy']
        if acc_diff > 0.01:
            f.write(f"✅ NEW is {acc_diff:.4f} points better\n")
        elif acc_diff < -0.01:
            f.write(f"❌ NEW is {abs(acc_diff):.4f} points worse\n")
        else:
            f.write(f"≈ Similar accuracy\n")
        
        # Convergence analysis
        f.write("\nCONVERGENCE ANALYSIS:\n")
        f.write("-" * 21 + "\n")
        
        for name, result in data.items():
            if result['convergence_generation'] is not None:
                f.write(f"{result['name']}: Converged at generation {result['convergence_generation']}\n")
            else:
                f.write(f"{result['name']}: No convergence detected\n")
        
        # Recommendations
        f.write("\nRECOMMENDATIONS:\n")
        f.write("-" * 16 + "\n")
        
        if speed_ratio > 2:
            f.write("⚠️  The new implementation is significantly slower.\n")
            f.write("   Consider optimizing the new architecture for better performance.\n")
        
        if acc_diff > 0.1:
            f.write("✅ The new implementation shows better accuracy.\n")
            f.write("   The modular architecture may be providing better optimization.\n")
        
        if old_result['final_accuracy'] < 0.5 and new_result['final_accuracy'] < 0.5:
            f.write("⚠️  Both implementations show low accuracy.\n")
            f.write("   Consider running longer tests or adjusting hyperparameters.\n")
    
    print(f"📄 Summary report saved to {report_path}")

def main():
    """Main visualization function."""
    print("📊 Loading comparison data...")
    data = load_comparison_data()
    
    # Create output directory
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    print("📈 Creating fitness curve plots...")
    plot_fitness_curves(data)
    
    print("📊 Creating performance comparison...")
    plot_performance_comparison(data)
    
    print("🎯 Creating convergence analysis...")
    plot_convergence_analysis(data)
    
    print("📄 Creating summary report...")
    create_summary_report(data)
    
    print("\n✅ All visualizations complete!")
    print(f"   Check the comparison_results/ directory for outputs")

if __name__ == "__main__":
    main()
