from config import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def save_structured_results(all_results, output_dir):
    """Save structured results in JSON and CSV formats"""
    # Save detailed JSON results
    json_path = os.path.join(output_dir, 'detailed_results.json')
        
    # Remove image features to reduce file size
    cleaned_results = {}
    for method_name, results in all_results.items():
        cleaned_results[method_name] = {
                k: v for k, v in results.items() 
                if k not in ['predictions', 'misclassifications']  # Remove large arrays
        }
        # Keep only summary of predictions
        if 'predictions' in results:
            cleaned_results[method_name]['prediction_summary'] = {
                'total_predictions': len(results['predictions']),
                'correct_predictions': results['correct_predictions'],
                'accuracy': results['accuracy']
            }
        
    with open(json_path, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    logging.info(f"Detailed results saved to: {json_path}")
        
    # Save summary CSV
    csv_path = os.path.join(output_dir, 'summary_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Type', 'Accuracy', 'Correct_Predictions', 
                        'Total_Images', 'Inference_Time', 'Templates_Used'])
            
        for method_name, results in all_results.items():
            method_type = 'baseline' if 'baseline' in method_name else 'ensemble'
            writer.writerow([
                method_name,
                method_type,
                f"{results['accuracy']:.4f}",
                results['correct_predictions'],
                results['total_images'],
                f"{results['inference_time']:.2f}",
                results['templates_used']
            ])
    logging.info(f"Summary results saved to: {csv_path}")

def generate_text_report(all_results, output_dir):
    """Generate a comprehensive text report with formatted tables"""
    report_path = os.path.join(output_dir, 'experiment_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLIP ZERO-SHOT CLASSIFICATION EXPERIMENT REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXPERIMENT SETUP\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: ViT-B/32\n")
        f.write(f"Dataset: Caltech101\n")
        f.write(f"Number of Classes: {len(CALTECH101_CLASSES)}\n")
        f.write(f"Simple Templates: {len(SIMPLE_TEMPLATES)}\n")
        f.write(f"Ensemble Templates: {len(ENSEMBLE_TEMPLATES)}\n\n")
        
        # Separate baseline and ensemble methods
        baseline_results = {k: v for k, v in all_results.items() if 'baseline' in k}
        ensemble_results = {k: v for k, v in all_results.items() if 'baseline' not in k}
        
        f.write("BASELINE COMPARISON (Feature Averaging Method)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Method':<20} {'Accuracy':<12} {'Templates':<10} {'Time(s)':<10}\n")
        f.write("-" * 50 + "\n")
        for method_name, results in baseline_results.items():
            f.write(f"{method_name:<20} {results['accuracy']:<12.4f} {results['templates_used']:<10} {results['inference_time']:<10.2f}\n")
        f.write("\n")
        
        f.write("ENSEMBLE IMPLEMENTATION METHODS COMPARISON\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Method':<20} {'Accuracy':<12} {'Time(s)':<10} {'Description':<40}\n")
        f.write("-" * 70 + "\n")
        for method_name, results in ensemble_results.items():
            f.write(f"{method_name:<20} {results['accuracy']:<12.4f} {results['inference_time']:<10.2f} {results['description'][:38]:<40}\n")
        f.write("\n")
        
        # Performance comparison
        f.write("OVERALL PERFORMANCE RANKING\n")
        f.write("-" * 60 + "\n")
        ranked_methods = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        f.write(f"{'Rank':<6} {'Method':<20} {'Accuracy':<12} {'Time(s)':<10} {'Type':<10}\n")
        f.write("-" * 60 + "\n")
        for i, (method_name, results) in enumerate(ranked_methods, 1):
            method_type = 'baseline' if 'baseline' in method_name else 'ensemble'
            f.write(f"{i:<6} {method_name:<20} {results['accuracy']:<12.4f} {results['inference_time']:<10.2f} {method_type:<10}\n")
        f.write("\n")
        
        # Improvement analysis
        if 'simple_baseline' in all_results and 'ensemble_baseline' in all_results:
            simple_acc = all_results['simple_baseline']['accuracy']
            ensemble_baseline_acc = all_results['ensemble_baseline']['accuracy']
            
            f.write("IMPROVEMENT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Simple → Ensemble Baseline: +{ensemble_baseline_acc - simple_acc:.4f} ")

# 自定义模板集合
CUSTOM_TEMPLATES = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of a {}",
    "a photograph of a {}",
    "a close-up photo of a {}",
    "a detailed photo of a {}",
    "a clear photo of a {}",
    "a high-quality photo of a {}",
    "a professional photo of a {}",
    "a beautiful photo of a {}",
    "a nice photo of a {}",
    "a good photo of a {}",
    "a great photo of a {}",
    "this is a {}",
    "this appears to be a {}",
    "this looks like a {}",
    "this resembles a {}",
    "a {} in the scene",
    "a {} in the image",
    "a {} in the photo",
    # 添加一些更具描述性的模板
    "a centered photo of a {}",
    "a well-lit photo of a {}",
    "a focused photo of a {}",
    "a sharp photo of a {}",
    "a colorful photo of a {}",
]


def main():
    """Main function to run all experiments - Simplified version"""
    # Initialize experiment
    experiment = CLIPExperiment("ViT-B/32")
    
    logging.info("Starting CALTECH101 zero-shot classification experiment")
    
    # Dataset path
    dataset_path = "./caltech-101"
    logging.info(f"Using dataset path: {dataset_path}")
    
    all_results = {}
    
    # 1. BASELINE EXPERIMENTS
    logging.info("=== RUNNING BASELINE EXPERIMENTS ===")
    
    # Simple template baseline
    simple_results = experiment.evaluate_baseline(
        dataset_path, SIMPLE_TEMPLATES, "simple_baseline", max_images_per_class=20
    )
    all_results['simple_baseline'] = simple_results
    
    # Ensemble template baseline
    ensemble_baseline_results = experiment.evaluate_baseline(
        dataset_path, ENSEMBLE_TEMPLATES, "ensemble_baseline", max_images_per_class=20
    )
    all_results['ensemble_baseline'] = ensemble_baseline_results
    
    # 2. ENSEMBLE IMPLEMENTATION EXPERIMENTS
    logging.info("=== RUNNING ENSEMBLE IMPLEMENTATION EXPERIMENTS ===")
    
    ensemble_methods = [
        ("feature_average", ENSEMBLE_TEMPLATES),
        ("prediction_average", ENSEMBLE_TEMPLATES),
        ("majority_voting", ENSEMBLE_TEMPLATES),
        ("weighted_prediction", ENSEMBLE_TEMPLATES),
        ("learned_weighting", ENSEMBLE_TEMPLATES),
        ("selective_ensemble", ENSEMBLE_TEMPLATES),
        ("random_forest_weighting", ENSEMBLE_TEMPLATES),
        ("variance_weighting", ENSEMBLE_TEMPLATES),
    ]
    
    for method_name, templates in ensemble_methods:
        if method_name == "selective_ensemble":
            results = experiment.selective_ensemble_fast(dataset_path, top_k=8, max_images_per_class=20)
        elif method_name == "random_forest_weighting":
            results = experiment.random_forest_weighting(dataset_path, templates, max_images_per_class=20)
        elif method_name == "variance_weighting":
            results = experiment.variance_weighting(dataset_path, templates, max_images_per_class=20)
        else:
            results = experiment.evaluate_ensemble_method(
                dataset_path, method_name, templates, max_images_per_class=20
            )
        all_results[method_name] = results
    
    # 3. 自定义模板实验
    logging.info("=== RUNNING CUSTOM TEMPLATES EXPERIMENT ===")
    custom_results = experiment.evaluate_baseline(
        dataset_path, CUSTOM_TEMPLATES, "custom_templates", max_images_per_class=20
    )
    all_results['custom_templates'] = custom_results
    
    # 4. 自定义模板的集成方法
    logging.info("=== RUNNING CUSTOM TEMPLATES ENSEMBLE ===")
    custom_ensemble_results = experiment.evaluate_ensemble_method(
        dataset_path, "feature_average", CUSTOM_TEMPLATES, "custom_ensemble", max_images_per_class=20
    )
    all_results['custom_ensemble'] = custom_ensemble_results
    
    # 创建输出目录和生成报告
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    generate_text_report(all_results, output_dir)
    save_structured_results(all_results, output_dir)
    
    # 最终总结
    logging.info("=" * 70)
    logging.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logging.info("=" * 70)
    
    # 打印结果排名
    logging.info("OVERALL RESULTS RANKING:")
    ranked_methods = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (method, results) in enumerate(ranked_methods, 1):
        logging.info(f"{i:2d}. {method:25} : {results['accuracy']:.4f}")
    
    # 打印随机森林权重信息（如果存在）
    if 'random_forest_weighting' in all_results:
        rf_results = all_results['random_forest_weighting']
        logging.info("\nRANDOM FOREST WEIGHT ANALYSIS:")
        if 'template_weights' in rf_results:
            weights = rf_results['template_weights']
            templates = ENSEMBLE_TEMPLATES[:len(weights)]
            
            # 显示权重分布
            logging.info(f"  Weight range: {min(weights):.4f} - {max(weights):.4f}")
            logging.info(f"  Weight std: {np.std(weights):.4f}")
            
            # 显示前5个最高权重的模板
            sorted_weights = sorted(zip(templates, weights), key=lambda x: x[1], reverse=True)
            logging.info("  Top 5 templates by weight:")
            for template, weight in sorted_weights[:5]:
                logging.info(f"    {weight:.4f}: {template}")
            
            # 显示前5个最低权重的模板
            logging.info("  Bottom 5 templates by weight:")
            for template, weight in sorted_weights[-5:]:
                logging.info(f"    {weight:.4f}: {template}")

if __name__ == "__main__":
    main()
    