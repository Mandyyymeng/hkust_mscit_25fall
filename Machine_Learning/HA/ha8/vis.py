import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from matplotlib.patches import Patch

# 使用普通的 matplotlib 样式，避免 LaTeX 依赖
plt.style.use('default')
sns.set_style("whitegrid")

class ExperimentVisualizer:
    def __init__(self, results_dir="experiment_results"):
        """Initialize visualizer with experiment results"""
        self.results_dir = results_dir
        self.results = self.load_results()
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
    def load_results(self):
        """Load experiment results from JSON file"""
        results_path = os.path.join(self.results_dir, 'detailed_results.json')
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def get_colors(self, n):
        """Dynamically generate n colors"""
        return plt.cm.Set3(np.linspace(0, 1, n))
    
    def plot_accuracy_comparison(self, ax):
        """Plot accuracy comparison across all methods"""
        methods = list(self.results.keys())
        accuracies = [self.results[method]['accuracy'] for method in methods]
        colors = self.get_colors(len(methods))
        
        bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_xlabel('Ensemble Method', fontweight='bold')
        ax.set_title('Accuracy Comparison Across Ensemble Methods', fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add method descriptions
        descriptions = [self.results[method].get('description', 'No description') for method in methods]
        ax.text(0.02, 0.98, 'Method Descriptions:', transform=ax.transAxes, 
               fontweight='bold', va='top', fontsize=9)
        for i, desc in enumerate(descriptions):
            ax.text(0.02, 0.90 - i*0.08, f'{i+1}. {desc}', transform=ax.transAxes,
                   va='top', fontsize=8, style='italic')
    
    def plot_inference_time(self, ax):
        """Plot inference time comparison"""
        methods = list(self.results.keys())
        times = [self.results[method]['inference_time'] for method in methods]
        colors = self.get_colors(len(methods))
        
        bars = ax.bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_ylabel('Inference Time (seconds)', fontweight='bold')
        ax.set_xlabel('Ensemble Method', fontweight='bold')
        ax.set_title('Computational Efficiency Comparison', fontweight='bold', pad=15)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_accuracy_vs_time(self, ax):
        """Plot accuracy vs inference time scatter plot"""
        methods = list(self.results.keys())
        accuracies = [self.results[method]['accuracy'] for method in methods]
        times = [self.results[method]['inference_time'] for method in methods]
        colors = self.get_colors(len(methods))
        
        scatter = ax.scatter(times, accuracies, c=colors, s=200, alpha=0.8, 
                           edgecolors='black', linewidths=1.5)
        
        # Add method labels
        for i, method in enumerate(methods):
            ax.annotate(method.replace('_', '\n'), (times[i], accuracies[i]),
                       xytext=(10, 10), textcoords='offset points',
                       fontweight='bold', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Inference Time (seconds)', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Accuracy vs Computational Cost', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        
        # Add quadrant annotations
        x_median = np.median(times)
        y_median = np.median(accuracies)
        ax.axhline(y=y_median, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=x_median, color='red', linestyle='--', alpha=0.5)
        
        ax.text(0.05, 0.95, 'Fast & Accurate', transform=ax.transAxes, 
               fontweight='bold', color='green', fontsize=10)
        ax.text(0.65, 0.95, 'Accurate but Slow', transform=ax.transAxes,
               fontweight='bold', color='orange', fontsize=10)
    
    def plot_per_class_heatmap(self, ax):
        """Plot heatmap of per-class accuracy for all methods"""
        methods = list(self.results.keys())
        
        # Get top 20 classes by average accuracy
        class_accuracies = {}
        for method_name, results in self.results.items():
            for class_name, accuracy in results['per_class_accuracy'].items():
                if class_name not in class_accuracies:
                    class_accuracies[class_name] = []
                class_accuracies[class_name].append(accuracy)
        
        # Calculate average accuracy per class
        avg_class_acc = {cls: np.mean(accs) for cls, accs in class_accuracies.items()}
        top_classes = sorted(avg_class_acc.items(), key=lambda x: x[1], reverse=True)[:20]
        top_class_names = [cls for cls, _ in top_classes]
        
        # Create heatmap data
        heatmap_data = []
        for class_name in top_class_names:
            row = []
            for method_name in methods:
                row.append(self.results[method_name]['per_class_accuracy'].get(class_name, 0))
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set labels
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45)
        ax.set_yticks(range(len(top_class_names)))
        ax.set_yticklabels(top_class_names, fontsize=8)
        
        # Add text annotations
        for i in range(len(top_class_names)):
            for j in range(len(methods)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=7, fontweight='bold')
        
        ax.set_title('Per-Class Accuracy Heatmap (Top 20 Classes)', fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, label='Accuracy')
    
    def plot_template_efficiency(self, ax):
        """Plot template efficiency analysis"""
        methods = list(self.results.keys())
        templates_used = [self.results[method]['templates_used'] for method in methods]
        accuracies = [self.results[method]['accuracy'] for method in methods]
        efficiency = [acc / time for acc, time in zip(accuracies, 
                     [self.results[method]['inference_time'] for method in methods])]
        
        # Create scatter plot with bubble sizes
        scatter = ax.scatter(templates_used, accuracies, s=[eff * 5000 for eff in efficiency],
                           c=efficiency, cmap='viridis', alpha=0.7, edgecolors='black')
        
        # Add method labels
        for i, method in enumerate(methods):
            ax.annotate(method.replace('_', '\n'), (templates_used[i], accuracies[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Number of Templates Used', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Template Efficiency Analysis\n(Bubble size = Accuracy/Time)', 
                    fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Efficiency (Accuracy/Time)')
    
    def plot_random_forest_analysis(self):
        """可视化随机森林权重分析结果"""
        if 'random_forest_weighting' not in self.results:
            print("No random forest weighting results found")
            return
        
        rf_results = self.results['random_forest_weighting']
        
        # 检查是否有权重分析数据
        if 'template_weights' not in rf_results:
            print("No template weights found in random forest results")
            return
        
        # 加载随机森林详细分析数据
        rf_analysis_path = os.path.join(self.results_dir, "random_forest_weight_analysis.json")
        if not os.path.exists(rf_analysis_path):
            print(f"Random forest analysis file not found: {rf_analysis_path}")
            return
        
        with open(rf_analysis_path, 'r') as f:
            rf_analysis = json.load(f)
        
        # 创建随机森林分析图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Random Forest Weight Analysis', fontsize=16, fontweight='bold')
        
        # 1. 模板权重条形图
        templates = list(rf_analysis['template_weights'].keys())
        weights = list(rf_analysis['template_weights'].values())
        
        # 截断长模板名称用于显示
        short_templates = [t[:30] + '...' if len(t) > 30 else t for t in templates]
        
        bars = axes[0, 0].bar(range(len(templates)), weights, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Template Weights Learned by Random Forest', fontweight='bold')
        axes[0, 0].set_xlabel('Templates')
        axes[0, 0].set_ylabel('Weight')
        axes[0, 0].set_xticks(range(len(templates)))
        axes[0, 0].set_xticklabels(short_templates, rotation=45, ha='right', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 在条形上添加数值
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                          f'{weight:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        # 2. 特征重要性分布
        axes[0, 1].hist(rf_analysis['feature_importances'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Distribution of Feature Importances', fontweight='bold')
        axes[0, 1].set_xlabel('Feature Importance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 权重与模板长度的关系（如果有模板数据）
        template_lengths = [len(t) for t in templates]
        axes[1, 0].scatter(template_lengths, weights, alpha=0.6, s=50, color='green')
        axes[1, 0].set_title('Template Length vs Learned Weight', fontweight='bold')
        axes[1, 0].set_xlabel('Template Length (characters)')
        axes[1, 0].set_ylabel('Learned Weight')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(template_lengths) > 1:
            z = np.polyfit(template_lengths, weights, 1)
            p = np.poly1d(z)
            axes[1, 0].plot(template_lengths, p(template_lengths), "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
            axes[1, 0].legend()
        
        # 4. 权重统计信息
        axes[1, 1].axis('off')
        stats_text = (
            f"Random Forest Weight Statistics:\n\n"
            f"Total Templates: {len(templates)}\n"
            f"Weight Range: {min(weights):.4f} - {max(weights):.4f}\n"
            f"Weight Std: {np.std(weights):.4f}\n"
            f"Mean Weight: {np.mean(weights):.4f}\n\n"
            f"Top 3 Templates:\n"
        )
        
        # 添加前3个模板
        sorted_weights = sorted(rf_analysis['template_weights'].items(), key=lambda x: x[1], reverse=True)
        for i, (template, weight) in enumerate(sorted_weights[:3]):
            stats_text += f"{i+1}. {weight:.4f}: {template[:40]}...\n"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.results_dir, "random_forest_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Random forest analysis plot saved to: random_forest_analysis.png")
        
        # 打印权重统计信息到控制台
        print("\nRandom Forest Weight Analysis:")
        print(f"  - Weight range: {min(weights):.4f} - {max(weights):.4f}")
        print(f"  - Weight std: {np.std(weights):.4f}")
        print(f"  - Top 5 templates by weight:")
        for template, weight in sorted_weights[:5]:
            print(f"    {weight:.4f}: {template}")
    
    def create_individual_plots(self):
        """Create individual high-quality plots"""
        individual_dir = os.path.join(self.results_dir, 'individual_plots')
        os.makedirs(individual_dir, exist_ok=True)
        
        # Create individual plots
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_accuracy_comparison(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_inference_time(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, 'inference_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_accuracy_vs_time(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, 'accuracy_vs_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        self.plot_per_class_heatmap(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, 'per_class_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_template_efficiency(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, 'template_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual plots saved to: {individual_dir}")
        
        # 创建随机森林分析图
        self.plot_random_forest_analysis()
    
    def generate_performance_summary(self):
        """Generate a text summary of performance metrics"""
        summary_path = os.path.join(self.results_dir, 'performance_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("CLIP Zero-Shot Classification Performance Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall performance
            f.write("OVERALL PERFORMANCE RANKING:\n")
            f.write("-" * 40 + "\n")
            ranked_methods = sorted(self.results.items(), 
                                  key=lambda x: x[1]['accuracy'], reverse=True)
            
            for i, (method_name, results) in enumerate(ranked_methods, 1):
                f.write(f"{i}. {method_name:<20} Accuracy: {results['accuracy']:.4f} "
                       f"Time: {results['inference_time']:.2f}s "
                       f"Efficiency: {results['accuracy']/results['inference_time']:.4f}\n")
            
            f.write("\n")
            
            # Best and worst performing classes
            best_method = ranked_methods[0][1]
            best_classes = sorted(best_method['per_class_accuracy'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]
            worst_classes = sorted(best_method['per_class_accuracy'].items(), 
                                 key=lambda x: x[1])[:10]
            
            f.write("BEST PERFORMING CLASSES:\n")
            f.write("-" * 40 + "\n")
            for class_name, accuracy in best_classes:
                f.write(f"{class_name:<30} {accuracy:.4f}\n")
            
            f.write("\nWORST PERFORMING CLASSES:\n")
            f.write("-" * 40 + "\n")
            for class_name, accuracy in worst_classes:
                f.write(f"{class_name:<30} {accuracy:.4f}\n")
            
            # 添加随机森林权重信息
            if 'random_forest_weighting' in self.results:
                rf_results = self.results['random_forest_weighting']
                if 'template_weights' in rf_results:
                    f.write("\nRANDOM FOREST WEIGHT ANALYSIS:\n")
                    f.write("-" * 40 + "\n")
                    weights = rf_results['template_weights']
                    templates = list(rf_results.get('template_weights_dict', {}).keys())
                    if not templates:
                        templates = [f"Template_{i}" for i in range(len(weights))]
                    
                    sorted_weights = sorted(zip(templates, weights), key=lambda x: x[1], reverse=True)
                    f.write("Top 5 templates by weight:\n")
                    for template, weight in sorted_weights[:5]:
                        f.write(f"  {weight:.4f}: {template}\n")
        
        print(f"Performance summary saved to: {summary_path}")

def main():
    """Main function to generate all visualizations"""
    print("Generating comprehensive visualizations...")
    res_folder = "experiment_results"
    
    # Initialize visualizer
    visualizer = ExperimentVisualizer(res_folder)
    # Create comprehensive visualization report
    visualizer.create_individual_plots()
    # Generate performance summary
    visualizer.generate_performance_summary()
    
    print("All visualizations completed successfully!")
    print(f"Results saved to: {visualizer.results_dir}")

if __name__ == "__main__":
    main()