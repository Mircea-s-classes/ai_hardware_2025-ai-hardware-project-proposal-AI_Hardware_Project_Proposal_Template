#!/usr/bin/env python3
"""
Generate comparison charts for MacBook vs Raspberry Pi 4
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'macbook': '#0071e3', 'rpi': '#e31c23'}

def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath) as f:
        data = json.load(f)
    return data['statistics']

def create_inference_comparison(macbook_stats, rpi_stats, output_dir):
    """Create inference time comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = ['MediaPipe', 'Random Forest', 'Total Pipeline']
    macbook_times = [
        macbook_stats['mediapipe_inference_ms']['mean'],
        macbook_stats['classifier_inference_ms']['mean'],
        macbook_stats['mediapipe_inference_ms']['mean'] + macbook_stats['classifier_inference_ms']['mean']
    ]
    rpi_times = [
        rpi_stats['mediapipe_inference_ms']['mean'],
        rpi_stats['classifier_inference_ms']['mean'],
        rpi_stats['mediapipe_inference_ms']['mean'] + rpi_stats['classifier_inference_ms']['mean']
    ]
    
    x = np.arange(len(components))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, macbook_times, width, label='MacBook', color=colors['macbook'], alpha=0.8)
    bars2 = ax.bar(x + width/2, rpi_times, width, label='Raspberry Pi 4', color=colors['rpi'], alpha=0.8)
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('AI Inference Latency Comparison\nMacBook vs Raspberry Pi 4', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}ms',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add speedup annotations
    for i, (mac, rpi) in enumerate(zip(macbook_times, rpi_times)):
        speedup = rpi / mac
        ax.text(i, max(mac, rpi) * 1.15, f'{speedup:.1f}x slower',
               ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Generated: inference_comparison.png")
    plt.close()

def create_fps_comparison(macbook_stats, rpi_stats, output_dir):
    """Create FPS comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall FPS comparison
    platforms = ['MacBook', 'Raspberry Pi 4']
    fps_values = [
        macbook_stats['average_fps_overall'],
        rpi_stats['average_fps_overall']
    ]
    colors_list = [colors['macbook'], colors['rpi']]
    
    bars = ax1.bar(platforms, fps_values, color=colors_list, alpha=0.8, width=0.6)
    ax1.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target (10 FPS)', alpha=0.7)
    ax1.set_ylabel('Frames Per Second', fontsize=12, fontweight='bold')
    ax1.set_title('Overall FPS Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(fps_values) * 1.3)
    
    # Add value labels
    for bar, val in zip(bars, fps_values):
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f} FPS',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add status icons
    for i, (bar, val) in enumerate(zip(bars, fps_values)):
        status = '✅' if val >= 10 else '⚠️'
        ax1.text(bar.get_x() + bar.get_width()/2., val * 0.5,
                status, ha='center', fontsize=24)
    
    # FPS distribution (mean, min, max)
    categories = ['Mean FPS', 'Min FPS', 'Max FPS']
    macbook_fps_dist = [
        macbook_stats['fps']['mean'],
        macbook_stats['fps']['min'],
        macbook_stats['fps']['max']
    ]
    rpi_fps_dist = [
        rpi_stats['fps']['mean'],
        rpi_stats['fps']['min'],
        rpi_stats['fps']['max']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x - width/2, macbook_fps_dist, width, label='MacBook', color=colors['macbook'], alpha=0.8)
    ax2.bar(x + width/2, rpi_fps_dist, width, label='Raspberry Pi 4', color=colors['rpi'], alpha=0.8)
    
    ax2.set_ylabel('Frames Per Second', fontsize=12, fontweight='bold')
    ax2.set_title('FPS Distribution', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fps_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Generated: fps_comparison.png")
    plt.close()

def create_speedup_chart(macbook_stats, rpi_stats, output_dir):
    """Create speedup/slowdown chart"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    metrics = [
        'MediaPipe\nInference',
        'Random Forest\nClassifier',
        'Frame\nProcessing',
        'Overall\nFPS'
    ]
    
    # Calculate speedups (MacBook as baseline = 1.0)
    speedups = [
        macbook_stats['mediapipe_inference_ms']['mean'] / rpi_stats['mediapipe_inference_ms']['mean'],
        macbook_stats['classifier_inference_ms']['mean'] / rpi_stats['classifier_inference_ms']['mean'],
        macbook_stats['frame_time_ms']['mean'] / rpi_stats['frame_time_ms']['mean'],
        macbook_stats['average_fps_overall'] / rpi_stats['average_fps_overall']
    ]
    
    bars = ax.barh(metrics, speedups, color=colors['rpi'], alpha=0.8)
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=2, label='MacBook Baseline', alpha=0.7)
    ax.set_xlabel('MacBook Performance / RPi Performance', fontsize=12, fontweight='bold')
    ax.set_title('Raspberry Pi 4 Relative Performance\n(MacBook as Baseline = 1.0)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(speedups) * 1.2)
    
    # Add value labels
    for bar, val in zip(bars, speedups):
        label_x = val + max(speedups) * 0.02
        if val < 1:
            label = f'{val:.2f}x (faster)'
            color = 'green'
        else:
            label = f'{val:.1f}x (slower)'
            color = 'red'
        ax.text(label_x, bar.get_y() + bar.get_height()/2., label,
               ha='left', va='center', fontsize=10, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Generated: speedup_comparison.png")
    plt.close()

def create_cost_performance_chart(macbook_stats, rpi_stats, output_dir):
    """Create cost-performance comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Cost assumptions
    macbook_cost = 1500
    rpi_cost = 50
    
    macbook_fps = macbook_stats['average_fps_overall']
    rpi_fps = rpi_stats['average_fps_overall']
    
    # FPS per $100
    macbook_value = (macbook_fps / macbook_cost) * 100
    rpi_value = (rpi_fps / rpi_cost) * 100
    
    platforms = ['MacBook\n($1,500)', 'Raspberry Pi 4\n($50)']
    values = [macbook_value, rpi_value]
    colors_list = [colors['macbook'], colors['rpi']]
    
    bars = ax.bar(platforms, values, color=colors_list, alpha=0.8, width=0.5)
    ax.set_ylabel('FPS per $100', fontsize=12, fontweight='bold')
    ax.set_title('Cost-Performance Analysis\nFPS per $100 Spent', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., val,
               f'{val:.1f}\nFPS/$100',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add winner annotation
    winner_improvement = rpi_value / macbook_value
    ax.text(0.5, max(values) * 1.15, f'RPi: {winner_improvement:.0f}x better value!',
           ha='center', fontsize=12, fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cost_performance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Generated: cost_performance.png")
    plt.close()

def create_summary_dashboard(macbook_stats, rpi_stats, output_dir):
    """Create comprehensive summary dashboard"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Inference latency comparison (top left, spans 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    components = ['MediaPipe', 'Random Forest']
    macbook_times = [
        macbook_stats['mediapipe_inference_ms']['mean'],
        macbook_stats['classifier_inference_ms']['mean']
    ]
    rpi_times = [
        rpi_stats['mediapipe_inference_ms']['mean'],
        rpi_stats['classifier_inference_ms']['mean']
    ]
    x = np.arange(len(components))
    width = 0.35
    ax1.bar(x - width/2, macbook_times, width, label='MacBook', color=colors['macbook'], alpha=0.8)
    ax1.bar(x + width/2, rpi_times, width, label='Raspberry Pi 4', color=colors['rpi'], alpha=0.8)
    ax1.set_ylabel('Latency (ms)', fontsize=10, fontweight='bold')
    ax1.set_title('AI Inference Latency', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. FPS comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    platforms = ['MacBook', 'RPi4']
    fps_values = [macbook_stats['average_fps_overall'], rpi_stats['average_fps_overall']]
    bars = ax2.bar(platforms, fps_values, color=[colors['macbook'], colors['rpi']], alpha=0.8)
    ax2.axhline(y=10, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('FPS', fontsize=10, fontweight='bold')
    ax2.set_title('Overall FPS', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, fps_values):
        ax2.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Frame processing time distribution (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    metrics = ['Mean', 'P50', 'P95']
    macbook_frame = [
        macbook_stats['frame_time_ms']['mean'],
        macbook_stats['frame_time_ms']['p50'],
        macbook_stats['frame_time_ms']['p95']
    ]
    rpi_frame = [
        rpi_stats['frame_time_ms']['mean'],
        rpi_stats['frame_time_ms']['p50'],
        rpi_stats['frame_time_ms']['p95']
    ]
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, macbook_frame, width, color=colors['macbook'], alpha=0.8)
    ax3.bar(x + width/2, rpi_frame, width, color=colors['rpi'], alpha=0.8)
    ax3.set_ylabel('Time (ms)', fontsize=10, fontweight='bold')
    ax3.set_title('Frame Processing Time', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Speedup factors (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    speedup_categories = ['MediaPipe', 'Classifier', 'FPS']
    speedup_vals = [
        macbook_stats['mediapipe_inference_ms']['mean'] / rpi_stats['mediapipe_inference_ms']['mean'],
        macbook_stats['classifier_inference_ms']['mean'] / rpi_stats['classifier_inference_ms']['mean'],
        macbook_stats['average_fps_overall'] / rpi_stats['average_fps_overall']
    ]
    bars = ax4.barh(speedup_categories, speedup_vals, color=colors['rpi'], alpha=0.8)
    ax4.axvline(x=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('MacBook/RPi Ratio', fontsize=10, fontweight='bold')
    ax4.set_title('Relative Performance', fontsize=11, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, speedup_vals):
        ax4.text(val + 0.2, bar.get_y() + bar.get_height()/2., f'{val:.1f}x',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 5. Cost-performance (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    macbook_value = (macbook_stats['average_fps_overall'] / 1500) * 100
    rpi_value = (rpi_stats['average_fps_overall'] / 50) * 100
    platforms = ['MacBook', 'RPi4']
    values = [macbook_value, rpi_value]
    ax5.bar(platforms, values, color=[colors['macbook'], colors['rpi']], alpha=0.8)
    ax5.set_ylabel('FPS per $100', fontsize=10, fontweight='bold')
    ax5.set_title('Cost Efficiency', fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for i, val in enumerate(values):
        ax5.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Key metrics table (bottom, spans all)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    table_data = [
        ['Metric', 'MacBook', 'Raspberry Pi 4', 'Ratio'],
        ['MediaPipe Latency', f"{macbook_stats['mediapipe_inference_ms']['mean']:.1f} ms", 
         f"{rpi_stats['mediapipe_inference_ms']['mean']:.1f} ms",
         f"{rpi_stats['mediapipe_inference_ms']['mean']/macbook_stats['mediapipe_inference_ms']['mean']:.1f}x"],
        ['Classifier Latency', f"{macbook_stats['classifier_inference_ms']['mean']:.2f} ms",
         f"{rpi_stats['classifier_inference_ms']['mean']:.2f} ms",
         f"{rpi_stats['classifier_inference_ms']['mean']/macbook_stats['classifier_inference_ms']['mean']:.1f}x"],
        ['Overall FPS', f"{macbook_stats['average_fps_overall']:.1f}",
         f"{rpi_stats['average_fps_overall']:.1f}",
         f"{macbook_stats['average_fps_overall']/rpi_stats['average_fps_overall']:.2f}x"],
        ['CPU Usage', f"{macbook_stats['cpu_percent']['mean']:.1f}%",
         'N/A', '-'],
        ['Hardware Cost', '$1,500', '$50', '30x'],
        ['FPS per $100', f"{macbook_value:.2f}", f"{rpi_value:.1f}", f"{rpi_value/macbook_value:.0f}x better"]
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    fig.suptitle('MacBook vs Raspberry Pi 4: Comprehensive Performance Comparison',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✅ Generated: summary_dashboard.png")
    plt.close()

def main():
    """Main function"""
    # File paths
    base_dir = Path(__file__).parent
    macbook_metrics = base_dir / 'results/macbook_results/metrics/metrics_20251217_215834.json'
    rpi_metrics = base_dir / 'results/rpi_results/headless_metrics/metrics_20251217_213125.json'
    output_dir = base_dir / 'results/comparison_charts'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Generating Platform Comparison Charts...")
    print(f"   MacBook metrics: {macbook_metrics.name}")
    print(f"   RPi4 metrics: {rpi_metrics.name}")
    print(f"   Output: {output_dir}")
    print()
    
    # Load metrics
    macbook_stats = load_metrics(macbook_metrics)
    rpi_stats = load_metrics(rpi_metrics)
    
    # Generate charts
    create_inference_comparison(macbook_stats, rpi_stats, output_dir)
    create_fps_comparison(macbook_stats, rpi_stats, output_dir)
    create_speedup_chart(macbook_stats, rpi_stats, output_dir)
    create_cost_performance_chart(macbook_stats, rpi_stats, output_dir)
    create_summary_dashboard(macbook_stats, rpi_stats, output_dir)
    
    print()
    print("✅ All charts generated successfully!")
    print(f"📁 Charts saved to: {output_dir}")
    print()
    print("Generated charts:")
    print("  1. inference_comparison.png  - AI inference latency comparison")
    print("  2. fps_comparison.png        - FPS and frame rate analysis")
    print("  3. speedup_comparison.png    - Relative performance metrics")
    print("  4. cost_performance.png      - Cost-efficiency analysis")
    print("  5. summary_dashboard.png     - Comprehensive overview")

if __name__ == '__main__':
    main()

