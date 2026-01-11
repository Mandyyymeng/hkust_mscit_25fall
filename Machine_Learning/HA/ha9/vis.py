"""综合测试和可视化分析工具 - 增强版"""
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from PIL import Image
import argparse
from scipy import linalg
from torchvision.models import inception_v3
import torch.nn.functional as F
import subprocess
import pandas as pd
import seaborn as sns
from pathlib import Path

# 添加路径以便导入配置
import sys
sys.path.append('.')

from gan_main import *

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EnhancedVisualizer:
    def __init__(self, model_dir='saved_models', output_dir='test_results'):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        setup_logging()
        
        # 创建子目录
        self.sample_dir = os.path.join(output_dir, 'samples')
        self.metric_dir = os.path.join(output_dir, 'metrics')
        self.comparison_dir = os.path.join(output_dir, 'comparisons')
        
        for dir_path in [self.sample_dir, self.metric_dir, self.comparison_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
    def run_evaluation(self, num_samples=800, force_rerun=False):
        """运行评估，如果结果文件存在则直接加载"""
        results_file = os.path.join(self.output_dir, 'detailed_results.json')
        
        if os.path.exists(results_file) and not force_rerun:
            logging.info("Found existing results file, loading...")
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logging.info(f"Successfully loaded {len(results)} experiment results")
        else:
            logging.info("Running model evaluation...")
            # 调用test.py进行批量评估
            try:
                cmd = [
                    'python', 'test.py', 
                    '--eval_all',
                    '--num_samples', str(num_samples),
                    '--model_dir', self.model_dir
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logging.info("Evaluation completed")
                else:
                    logging.error(f"Evaluation failed: {result.stderr}")
                    return {}
                    
                # 加载评估结果
                if os.path.exists('fid_evaluation_results.json'):
                    with open('fid_evaluation_results.json', 'r', encoding='utf-8') as f:
                        results = json.load(f)
                else:
                    results = {}
                    
            except Exception as e:
                logging.error(f"Error running evaluation: {e}")
                results = {}
        
        return results

    def generate_comprehensive_visualizations(self, results):
        """生成全面的可视化"""
        if not results:
            logging.warning("No result data available for visualization")
            return
            
        # 1. 主要指标对比图
        self._plot_main_metrics(results)
        
        # 2. 配置参数影响分析
        self._plot_config_analysis(results)
        
        # 3. 样本质量展示
        self._generate_sample_showcase(results)
        
        # 4. 训练稳定性分析
        self._plot_training_stability(results)
        
        # 5. 生成详细报告
        self._generate_detailed_report(results)

    def _plot_main_metrics(self, results):
        """绘制主要指标对比"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 准备数据
        exp_names = list(results.keys())
        fid_scores = [results[exp]['fid_score'] for exp in exp_names]
        
        # 1. FID分数对比（柱状图）
        colors = ['lightcoral' if score > min(fid_scores) * 1.5 else 'lightseagreen' 
                 for score in fid_scores]
        
        bars = ax1.bar(exp_names, fid_scores, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('FID Score Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('FID Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars, fid_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 2. FID分数对比（散点图，按隐藏层大小着色）
        hidden_sizes = [results[exp]['config'].get('hidden_size', 256) for exp in exp_names]
        scatter = ax2.scatter(range(len(exp_names)), fid_scores, c=hidden_sizes, 
                            cmap='viridis', s=100, alpha=0.7)
        ax2.set_title('FID Score Distribution (Colored by Hidden Size)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('FID Score')
        ax2.set_xlabel('Experiment Index')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Hidden Size')
        
        # 3. 配置参数热力图
        if len(exp_names) > 1:
            config_features = []
            for exp in exp_names:
                config = results[exp]['config']
                features = {
                    'hidden_size': config.get('hidden_size', 256),
                    'latent_size': config.get('latent_size', 128),
                    'lr': config.get('lr', 0.0002),
                    'n_critic': config.get('n_critic', 1),
                }
                config_features.append(features)
            
            df = pd.DataFrame(config_features, index=exp_names)
            # 归一化
            df_normalized = (df - df.min()) / (df.max() - df.min())
            
            im = ax3.imshow(df_normalized.T, cmap='YlOrRd', aspect='auto')
            ax3.set_title('Configuration Parameters Heatmap', fontsize=14, fontweight='bold')
            ax3.set_xticks(range(len(exp_names)))
            ax3.set_xticklabels(exp_names, rotation=45)
            ax3.set_yticks(range(len(df_normalized.columns)))
            ax3.set_yticklabels(df_normalized.columns)
            plt.colorbar(im, ax=ax3)
        
        # 4. 性能排名
        sorted_results = sorted(results.items(), key=lambda x: x[1]['fid_score'])
        top_exps = [x[0] for x in sorted_results[:5]]
        top_fids = [x[1]['fid_score'] for x in sorted_results[:5]]
        
        bars = ax4.barh(top_exps, top_fids, color='lightblue', alpha=0.7)
        ax4.set_title('Top 5 Best Performing Models', fontsize=14, fontweight='bold')
        ax4.set_xlabel('FID Score')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars, top_fids):
            ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{score:.2f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, 'main_metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_config_analysis(self, results):
        """配置参数影响分析"""
        if len(results) < 3:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # 准备数据
        data = []
        for exp_name, result in results.items():
            config = result['config']
            data.append({
                'exp_name': exp_name,
                'fid_score': result['fid_score'],
                'hidden_size': config.get('hidden_size', 256),
                'latent_size': config.get('latent_size', 128),
                'lr': config.get('lr', 0.0002),
                'n_critic': config.get('n_critic', 1),
                'use_gp': config.get('use_gradient_penalty', False),
                'use_sn': config.get('use_spectral_norm', False),
            })
        
        df = pd.DataFrame(data)
        
        # 1. 隐藏层大小 vs FID
        axes[0].scatter(df['hidden_size'], df['fid_score'], alpha=0.7, s=100)
        axes[0].set_xlabel('Hidden Size')
        axes[0].set_ylabel('FID Score')
        axes[0].set_title('Impact of Hidden Size on FID')
        axes[0].grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(df) > 1:
            z = np.polyfit(df['hidden_size'], df['fid_score'], 1)
            p = np.poly1d(z)
            axes[0].plot(df['hidden_size'], p(df['hidden_size']), "r--", alpha=0.8)
        
        # 2. 学习率 vs FID
        axes[1].scatter(df['lr'], df['fid_score'], alpha=0.7, s=100, color='orange')
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('FID Score')
        axes[1].set_title('Impact of Learning Rate on FID')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 隐变量维度 vs FID
        axes[2].scatter(df['latent_size'], df['fid_score'], alpha=0.7, s=100, color='green')
        axes[2].set_xlabel('Latent Size')
        axes[2].set_ylabel('FID Score')
        axes[2].set_title('Impact of Latent Size on FID')
        axes[2].grid(True, alpha=0.3)
        
        # 4. 稳定技术效果比较
        gp_scores = df[df['use_gp']]['fid_score']
        sn_scores = df[df['use_sn']]['fid_score']
        baseline_scores = df[~(df['use_gp'] | df['use_sn'])]['fid_score']
        
        tech_data = []
        if len(gp_scores) > 0:
            tech_data.append(('Gradient\nPenalty', gp_scores.mean()))
        if len(sn_scores) > 0:
            tech_data.append(('Spectral\nNorm', sn_scores.mean()))
        if len(baseline_scores) > 0:
            tech_data.append(('Baseline', baseline_scores.mean()))
            
        if tech_data:
            tech_names, tech_means = zip(*tech_data)
            bars = axes[3].bar(tech_names, tech_means, alpha=0.7, color=['red', 'blue', 'gray'])
            axes[3].set_title('Stability Techniques Comparison')
            axes[3].set_ylabel('Average FID Score')
            axes[3].grid(True, alpha=0.3)
            
            for bar, mean in zip(bars, tech_means):
                axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{mean:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, 'config_impact_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_sample_showcase(self, results):
        """生成样本展示"""
        # 选择几个代表性模型生成样本
        if not results:
            return
            
        # 选择FID最好、最差和中间的模型
        sorted_results = sorted(results.items(), key=lambda x: x[1]['fid_score'])
        selected_exps = [
            sorted_results[0][0],  # 最佳
            sorted_results[len(sorted_results)//2][0],  # 中等
            sorted_results[-1][0]   # 最差
        ]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        for idx, exp_name in enumerate(selected_exps):
            if idx >= len(axes):
                break
                
            result = results[exp_name]
            config = result['config']
            
            # 加载模型生成样本
            try:
                model_path = result['model_path']
                generator = self._load_generator(model_path, config)
                
                with torch.no_grad():
                    z = torch.randn(16, config['latent_size']).to(device)
                    samples = generator(z).reshape(-1, 1, 28, 28)
                    grid = make_grid(denorm(samples), nrow=8, padding=2)
                
                axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap='gray')
                axes[idx].set_title(
                    f'{exp_name}\nFID: {result["fid_score"]:.2f}, '
                    f'Hidden: {config.get("hidden_size", 256)}', 
                    fontsize=12
                )
                axes[idx].axis('off')
                
            except Exception as e:
                logging.warning(f"Cannot generate samples for {exp_name}: {e}")
                axes[idx].text(0.5, 0.5, f'Cannot load\n{exp_name}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.sample_dir, 'model_comparison_samples.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_training_stability(self, results):
        """训练稳定性分析"""
        if len(results) < 2:
            return
            
        # 分析不同配置的训练稳定性
        stability_data = []
        
        for exp_name, result in results.items():
            config = result['config']
            stability_data.append({
                'exp_name': exp_name,
                'fid_score': result['fid_score'],
                'hidden_size': config.get('hidden_size', 256),
                'lr': config.get('lr', 0.0002),
                'n_critic': config.get('n_critic', 1),
                'has_stability_tech': config.get('use_gradient_penalty', False) or 
                                    config.get('use_spectral_norm', False)
            })
        
        df = pd.DataFrame(stability_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 稳定性技术 vs FID
        if 'has_stability_tech' in df.columns:
            stable_group = df[df['has_stability_tech']]
            unstable_group = df[~df['has_stability_tech']]
            
            groups = []
            means = []
            stds = []
            
            if len(stable_group) > 0:
                groups.append('With Stability\nTechniques')
                means.append(stable_group['fid_score'].mean())
                stds.append(stable_group['fid_score'].std())
                
            if len(unstable_group) > 0:
                groups.append('Without Stability\nTechniques')
                means.append(unstable_group['fid_score'].mean())
                stds.append(unstable_group['fid_score'].std())
            
            if groups:
                bars = axes[0].bar(groups, means, yerr=stds, capsize=5, 
                                 alpha=0.7, color=['lightgreen', 'lightcoral'])
                axes[0].set_title('Impact of Stability Techniques')
                axes[0].set_ylabel('Average FID Score')
                axes[0].grid(True, alpha=0.3)
        
        # 2. n_critic vs FID
        if 'n_critic' in df.columns:
            for n_critic in df['n_critic'].unique():
                subset = df[df['n_critic'] == n_critic]
                axes[1].scatter([n_critic] * len(subset), subset['fid_score'], 
                              alpha=0.6, s=80, label=f'n_critic={n_critic}')
            
            axes[1].set_xlabel('n_critic')
            axes[1].set_ylabel('FID Score')
            axes[1].set_title('Impact of n_critic on FID')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.metric_dir, 'training_stability_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_detailed_report(self, results):
        """生成详细报告"""
        if not results:
            return
            
        # 生成HTML报告
        html_content = self._generate_html_report(results)
        
        with open(os.path.join(self.output_dir, 'detailed_report.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 生成CSV文件用于进一步分析
        self._generate_csv_report(results)
        
        logging.info(f"Detailed report generated in: {self.output_dir}")

    def _generate_html_report(self, results):
        """生成HTML格式的详细报告"""
        sorted_results = sorted(results.items(), key=lambda x: x[1]['fid_score'])
        current_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>GAN Experiment Detailed Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric-card {{ background: white; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; }}
                .best {{ background: #d4edda; border-color: #c3e6cb; }}
                .worst {{ background: #f8d7da; border-color: #f5c6cb; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GAN Model Experiment Detailed Report</h1>
                <p>Generated: {current_time}</p>
                <p>Total Experiments: {len(results)}</p>
            </div>
        """
        
        # 最佳模型
        if sorted_results:
            best_exp, best_result = sorted_results[0]
            html += f"""
            <div class="metric-card best">
                <h2>🎉 Best Model: {best_exp}</h2>
                <p><strong>FID Score:</strong> {best_result['fid_score']:.4f}</p>
                <p><strong>Configuration:</strong> hidden_size={best_result['config'].get('hidden_size', 256)}, 
                latent_size={best_result['config'].get('latent_size', 128)}, 
                lr={best_result['config'].get('lr', 0.0002)}</p>
            </div>
            """
        
        # 详细结果表格
        html += """
            <div class="metric-card">
                <h2>📊 Detailed Experiment Results</h2>
                <table>
                    <tr>
                        <th>Experiment Name</th>
                        <th>FID Score</th>
                        <th>Hidden Size</th>
                        <th>Latent Size</th>
                        <th>Learning Rate</th>
                        <th>n_critic</th>
                        <th>Stability Techniques</th>
                    </tr>
        """
        
        for exp_name, result in sorted_results:
            config = result['config']
            stability_tech = []
            if config.get('use_gradient_penalty', False):
                stability_tech.append('GP')
            if config.get('use_spectral_norm', False):
                stability_tech.append('SN')
            
            html += f"""
                    <tr>
                        <td>{exp_name}</td>
                        <td>{result['fid_score']:.4f}</td>
                        <td>{config.get('hidden_size', 256)}</td>
                        <td>{config.get('latent_size', 128)}</td>
                        <td>{config.get('lr', 0.0002)}</td>
                        <td>{config.get('n_critic', 1)}</td>
                        <td>{', '.join(stability_tech) if stability_tech else 'None'}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="metric-card">
                <h2>📈 Visualization Charts</h2>
                <p>The following charts have been generated in the test_results/ directory:</p>
                <ul>
                    <li>main_metrics_comparison.png - Main metrics comparison</li>
                    <li>config_impact_analysis.png - Configuration parameter impact analysis</li>
                    <li>model_comparison_samples.png - Model sample comparison</li>
                    <li>training_stability_analysis.png - Training stability analysis</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html

    def _generate_csv_report(self, results):
        """生成CSV报告"""
        data = []
        for exp_name, result in results.items():
            config = result['config']
            row = {
                'experiment': exp_name,
                'fid_score': result['fid_score'],
                'hidden_size': config.get('hidden_size', 256),
                'latent_size': config.get('latent_size', 128),
                'lr': config.get('lr', 0.0002),
                'n_critic': config.get('n_critic', 1),
                'use_gradient_penalty': config.get('use_gradient_penalty', False),
                'use_spectral_norm': config.get('use_spectral_norm', False),
                'loss_function': config.get('loss_function', 'bce'),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, 'experiment_results.csv'), index=False, encoding='utf-8')

    def _load_generator(self, model_path, config):
        """加载生成器"""
        generator = Generator(
            latent_size=config['latent_size'],
            hidden_size=config['hidden_size'],
            image_size=config['image_size'],
            activation=config.get('activation', 'relu')
        ).to(device)
        
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()
        return generator

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GAN Model Visualization Analysis Tool')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='Model directory')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=800, help='Number of evaluation samples')
    parser.add_argument('--force_rerun', action='store_true', help='Force re-run evaluation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GAN Model Comprehensive Visualization Analysis Tool")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = EnhancedVisualizer(args.model_dir, args.output_dir)
    
    # 运行评估（如果结果不存在或强制重新运行）
    print("Loading or running model evaluation...")
    results = visualizer.run_evaluation(args.num_samples, args.force_rerun)
    
    if results:
        print(f"Successfully processed {len(results)} experiments")
        print("Generating visualizations...")
        
        # 生成可视化
        visualizer.generate_comprehensive_visualizations(results)
        
        print("✅ Visualization completed!")
        print(f"📁 Results saved in: {args.output_dir}/")
        print("📊 Includes:")
        print("   - Main metrics comparison charts")
        print("   - Configuration impact analysis") 
        print("   - Model sample showcase")
        print("   - Training stability analysis")
        print("   - Detailed HTML report")
        print("   - CSV data file")
    else:
        print("❌ No experimental results found")
        print("Please check:")
        print("1. saved_models/ directory exists")
        print("2. There are _G_final.pth model files")
        print("3. Try using --force_rerun parameter to re-run evaluation")

if __name__ == "__main__":
    main()