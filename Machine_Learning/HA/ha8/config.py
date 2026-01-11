import torch
import clip
import logging
import json
import os
import csv
from PIL import Image
import numpy as np
import time
from collections import Counter
from tqdm import tqdm
from prompts import CALTECH101_CLASSES, SIMPLE_TEMPLATES, ENSEMBLE_TEMPLATES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)

class CLIPExperiment:
    def __init__(self, model_name="ViT-B/32"):
        """Initialize CLIP experiment with specified model"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing CLIP experiment with model: {model_name}")
        logging.info(f"Using device: {self.device}")
        
        # Load CLIP model and preprocessing
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        logging.info("CLIP model loaded successfully")
    
    def find_matching_class_folder(self, class_folders, target_class):
        """Find matching folder name for target class based on actual Caltech101 structure"""
        
        # Manual mapping based on actual Caltech101 folder names
        manual_mapping = {
            'background': 'BACKGROUND_Google',
            'off-center face': 'Faces_easy',
            'centered face': 'Faces_easy', 
            'leopard': 'Leopards',
            'motorbike': 'Motorbikes',
            'accordion': 'accordion',
            'airplane': 'Airplanes_Side_2',
            'anchor': 'anchor',
            'ant': 'ant',
            'barrel': 'barrel',
            'bass': 'bass',
            'beaver': 'beaver',
            'binocular': 'binocular',
            'bonsai': 'bonsai',
            'brain': 'brain',
            'brontosaurus': 'brontosaurus',
            'buddha': 'buddha',
            'butterfly': 'butterfly',
            'camera': 'camera',
            'cannon': 'cannon',
            'side of a car': 'car_side',
            'ceiling fan': 'ceiling_fan',
            'cellphone': 'cellphone',
            'chair': 'chair',
            'chandelier': 'chandelier',
            'body of a cougar cat': 'cougar_body',
            'face of a cougar cat': 'cougar_face',
            'crab': 'crab',
            'crayfish': 'crayfish',
            'crocodile': 'crocodile',
            'head of a crocodile': 'crocodile_head',
            'cup': 'cup',
            'dalmatian': 'dalmatian',
            'dollar bill': 'dollar_bill',
            'dolphin': 'dolphin',
            'dragonfly': 'dragonfly',
            'electric guitar': 'electric_guitar',
            'elephant': 'elephant',
            'emu': 'emu',
            'euphonium': 'euphonium',
            'ewer': 'ewer',
            'ferry': 'ferry',
            'flamingo': 'flamingo',
            'head of a flamingo': 'flamingo_head',
            'garfield': 'garfield',
            'gerenuk': 'gerenuk',
            'gramophone': 'gramophone',
            'grand piano': 'grand_piano',
            'hawksbill': 'hawksbill',
            'headphone': 'headphone',
            'hedgehog': 'hedgehog',
            'helicopter': 'helicopter',
            'ibis': 'ibis',
            'inline skate': 'inline_skate',
            'joshua tree': 'joshua_tree',
            'kangaroo': 'kangaroo',
            'ketch': 'ketch',
            'lamp': 'lamp',
            'laptop': 'laptop',
            'llama': 'llama',
            'lobster': 'lobster',
            'lotus': 'lotus',
            'mandolin': 'mandolin',
            'mayfly': 'mayfly',
            'menorah': 'menorah',
            'metronome': 'metronome',
            'minaret': 'minaret',
            'nautilus': 'nautilus',
            'octopus': 'octopus',
            'okapi': 'okapi',
            'pagoda': 'pagoda',
            'panda': 'panda',
            'pigeon': 'pigeon',
            'pizza': 'pizza',
            'platypus': 'platypus',
            'pyramid': 'pyramid',
            'revolver': 'revolver',
            'rhino': 'rhino',
            'rooster': 'rooster',
            'saxophone': 'saxophone',
            'schooner': 'schooner',
            'scissors': 'scissors',
            'scorpion': 'scorpion',
            'sea horse': 'sea_horse',
            'snoopy (cartoon beagle)': 'snoopy',
            'soccer ball': 'soccer_ball',
            'stapler': 'stapler',
            'starfish': 'starfish',
            'stegosaurus': 'stegosaurus',
            'stop sign': 'stop_sign',
            'strawberry': 'strawberry',
            'sunflower': 'sunflower',
            'tick': 'tick',
            'trilobite': 'trilobite',
            'umbrella': 'umbrella',
            'watch': 'watch',
            'water lilly': 'water_lilly',
            'wheelchair': 'wheelchair',
            'wild cat': 'wild_cat',
            'windsor chair': 'windsor_chair',
            'wrench': 'wrench',
            'yin and yang symbol': 'yin_yang'
        }
        
        # Check manual mapping first
        if target_class in manual_mapping:
            mapped_folder = manual_mapping[target_class]
            if mapped_folder in class_folders:
                return mapped_folder
            else:
                logging.warning(f"Manual mapping '{target_class}' -> '{mapped_folder}' but folder not found")
        
        # Fallback: try fuzzy matching
        target_lower = target_class.lower()
        for folder in class_folders:
            folder_lower = folder.lower()
            
            # Exact match
            if folder_lower == target_lower:
                return folder
            
            # Contains match
            if target_lower in folder_lower or folder_lower in target_lower:
                return folder
            
            # Replace spaces with underscores and try again
            target_underscore = target_lower.replace(' ', '_')
            if target_underscore == folder_lower:
                return folder
        
        logging.warning(f"No matching folder found for class: '{target_class}'")
        return None
    
    def create_text_features_baseline(self, classes, templates):
        """
        Baseline method: Feature averaging (official CLIP approach)
        Returns: tensor of shape [num_classes, feature_dim]
        """
        text_features_per_class = []
        
        for class_name in tqdm(classes, desc="Processing classes", disable=True):
            # Create texts for all templates of this class
            texts = [template.format(class_name) for template in templates]
            text_tokens = clip.tokenize(texts).to(self.device)
            
            with torch.no_grad():
                # Encode all templates for this class [num_templates, feature_dim]
                class_embeddings = self.model.encode_text(text_tokens)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                
                # Average across templates to get single feature vector per class
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                
                text_features_per_class.append(class_embedding)
        
        # Stack to [num_classes, feature_dim]
        text_features = torch.stack(text_features_per_class, dim=0)
        return text_features
    
    def method1_feature_average(self, classes, templates):
        """
        Method 1: Feature Averaging (same as baseline but included for consistency)
        """
        return self.create_text_features_baseline(classes, templates)
    
    def method2_prediction_average(self, image_path, classes, templates):
        """
        Method 2: Prediction Averaging
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                all_similarities = []
                
                for template in templates:
                    text_inputs = [template.format(class_name) for class_name in classes]
                    text_tokens = clip.tokenize(text_inputs).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (100.0 * image_features @ text_features.T)
                    all_similarities.append(similarity)
                
                ensemble_similarity = torch.mean(torch.cat(all_similarities), dim=0)
                ensemble_similarity = ensemble_similarity.softmax(dim=-1)
                
                values, indices = ensemble_similarity.topk(5)
                
                predictions = []
                for value, idx in zip(values, indices):
                    if idx.item() < len(classes):
                        predictions.append({
                            'class': classes[idx.item()],
                            'confidence': value.item()
                        })
                
                return predictions
                
        except Exception as e:
            logging.error(f"Error in method2_prediction_average for {image_path}: {str(e)}")
            return None
    
    def method3_majority_voting(self, image_path, classes, templates):
        """
        Method 3: Majority Voting
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                votes = []
                confidence_scores = []
                
                for template in templates:
                    text_inputs = [template.format(class_name) for class_name in classes]
                    text_tokens = clip.tokenize(text_inputs).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    top_conf, top_idx = similarity[0].topk(1)
                    
                    votes.append(top_idx.item())
                    confidence_scores.append(top_conf.item())
                
                vote_counter = Counter(votes)
                most_common_vote = vote_counter.most_common(1)[0]
                
                winning_class_idx = most_common_vote[0]
                vote_count = most_common_vote[1]
                
                winning_votes_indices = [i for i, vote in enumerate(votes) if vote == winning_class_idx]
                avg_confidence = np.mean([confidence_scores[i] for i in winning_votes_indices])
                
                predictions = [{
                    'class': classes[winning_class_idx],
                    'confidence': avg_confidence,
                    'vote_count': vote_count,
                    'total_templates': len(templates)
                }]
                
                return predictions
                
        except Exception as e:
            logging.error(f"Error in method3_majority_voting for {image_path}: {str(e)}")
            return None
    
    def method4_weighted_prediction(self, image_path, classes, templates, weights=None):
        """
        Method 4: Weighted Prediction
        """
        if weights is None:
            weights = [1.0 / (i + 1) for i in range(len(templates))]
            weights = [w / sum(weights) for w in weights]
        
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                weighted_similarities = []
                
                for i, template in enumerate(templates):
                    text_inputs = [template.format(class_name) for class_name in classes]
                    text_tokens = clip.tokenize(text_inputs).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (100.0 * image_features @ text_features.T)
                    weighted_similarity = similarity * weights[i]
                    weighted_similarities.append(weighted_similarity)
                
                final_similarity = torch.sum(torch.cat(weighted_similarities), dim=0)
                final_similarity = final_similarity.softmax(dim=-1)
                
                values, indices = final_similarity.topk(5)
                
                predictions = []
                for value, idx in zip(values, indices):
                    if idx.item() < len(classes):
                        predictions.append({
                            'class': classes[idx.item()],
                            'confidence': value.item()
                        })
                
                return predictions
                
        except Exception as e:
            logging.error(f"Error in method4_weighted_prediction for {image_path}: {str(e)}")
            return None

    def method5_learned_weighting(self, image_path, classes, templates, validation_ratio=0.2):
        """
        Method 5: Learned Weighting - 使用逻辑回归学习模板权重
        """
        try:
            # 第一步：在小规模验证集上学习模板权重
            if not hasattr(self, 'template_weights'):
                self._learn_template_weights(classes, templates, validation_ratio)
            
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                weighted_similarities = []
                
                for i, template in enumerate(templates):
                    text_inputs = [template.format(class_name) for class_name in classes]
                    text_tokens = clip.tokenize(text_inputs).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (100.0 * image_features @ text_features.T)
                    weighted_similarity = similarity * self.template_weights[i]
                    weighted_similarities.append(weighted_similarity)
                
                final_similarity = torch.sum(torch.cat(weighted_similarities), dim=0)
                final_similarity = final_similarity.softmax(dim=-1)
                
                values, indices = final_similarity.topk(5)
                
                predictions = []
                for value, idx in zip(values, indices):
                    if idx.item() < len(classes):
                        predictions.append({
                            'class': classes[idx.item()],
                            'confidence': value.item()
                        })
                
                return predictions
                
        except Exception as e:
            logging.error(f"Error in method5_learned_weighting for {image_path}: {str(e)}")
            return None
    def random_forest_weighting(self, dataset_path, templates, max_images_per_class=20):
        """使用随机森林学习模板权重，并保存权重与预测结果的关系"""
        logging.info(f"Running Random Forest Weighting with {len(templates)} templates")
        start_time = time.time()
        
        # 获取图像和标签
        images, true_labels = self.load_dataset(dataset_path, max_images_per_class)
        
        # 为每个模板生成预测概率
        all_template_probs = []
        for i, template in enumerate(templates):
            logging.info(f"Processing template {i+1}/{len(templates)}: {template}")
            text_features = self.get_text_features([template.format(cls) for cls in CALTECH101_CLASSES])
            probs = self.compute_probabilities(images, text_features)
            all_template_probs.append(probs)
        
        # 准备训练数据（使用一部分数据进行训练）
        n_train = len(images) // 2
        X_train = np.concatenate([prob[:n_train].reshape(n_train, -1) for prob in all_template_probs], axis=1)
        y_train = true_labels[:n_train]
        
        # 训练随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # 使用特征重要性作为权重
        feature_importances = rf.feature_importances_
        template_weights = []
        n_classes = len(CALTECH101_CLASSES)
        
        # 计算每个模板的权重
        for i in range(len(templates)):
            start_idx = i * n_classes
            end_idx = (i + 1) * n_classes
            template_weight = np.mean(feature_importances[start_idx:end_idx])
            template_weights.append(template_weight)
        
        # 归一化权重
        template_weights = np.array(template_weights)
        template_weights = template_weights / np.sum(template_weights)
        
        # 应用权重进行预测
        final_predictions = []
        final_confidences = []
        weighted_probs_all = []  # 保存每个样本的加权概率
        
        for i in range(len(images)):
            weighted_probs = np.zeros(len(CALTECH101_CLASSES))
            
            for j, weight in enumerate(template_weights):
                weighted_probs += weight * all_template_probs[j][i]
            
            weighted_probs_all.append(weighted_probs)
            predicted_class_idx = np.argmax(weighted_probs)
            confidence = weighted_probs[predicted_class_idx]
            
            final_predictions.append(predicted_class_idx)
            final_confidences.append(confidence)
        
        # 计算准确率
        correct_predictions = sum(1 for i in range(len(images)) 
                                if final_predictions[i] == true_labels[i])
        accuracy = correct_predictions / len(images)
        
        # 保存权重与预测结果的关系数据
        weight_prediction_relationship = {
            'template_weights': dict(zip(templates, template_weights.tolist())),
            'feature_importances': feature_importances.tolist(),
            'weighted_predictions_sample': weighted_probs_all[:10],  # 保存前10个样本的加权概率
            'template_predictions_sample': [probs[:10] for probs in all_template_probs],  # 每个模板的前10个预测
            'true_labels_sample': true_labels[:10],
            'final_predictions_sample': final_predictions[:10],
            'rf_feature_importance_by_class': {}
        }
        
        # 分析每个类别的特征重要性
        for class_idx in range(len(CALTECH101_CLASSES)):
            class_importances = []
            for i in range(len(templates)):
                start_idx = i * n_classes + class_idx
                class_importances.append(feature_importances[start_idx])
            weight_prediction_relationship['rf_feature_importance_by_class'][CALTECH101_CLASSES[class_idx]] = class_importances
        
        # 保存到文件
        output_dir = "experiment_results"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "random_forest_weight_analysis.json"), 'w') as f:
            json.dump(weight_prediction_relationship, f, indent=2)
        
        end_time = time.time()
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_images': len(images),
            'inference_time': end_time - start_time,
            'templates_used': len(templates),
            'description': 'Random Forest learned template weights based on feature importance',
            'template_weights': template_weights.tolist(),
            'per_class_accuracy': self.calculate_per_class_accuracy(true_labels, final_predictions, CALTECH101_CLASSES),
            'weight_analysis': weight_prediction_relationship  # 包含权重与预测关系的详细数据
        }
    
    def evaluate_baseline(self, dataset_path, templates, method_name, max_images_per_class=20):
        """Evaluate baseline methods using feature averaging approach"""
        logging.info(f"Evaluating baseline: {method_name}")
        
        results = {
            'method_name': method_name,
            'description': f'Baseline with {len(templates)} template(s) using feature averaging',
            'correct_predictions': 0,
            'total_images': 0,
            'accuracy': 0.0,
            'inference_time': 0.0,
            'templates_used': len(templates),
            'per_class_accuracy': {},
            'predictions': []
        }
        
        categories_path = os.path.join(dataset_path, "101_ObjectCategories")
        
        if not os.path.exists(categories_path):
            logging.error(f"Dataset directory not found: {categories_path}")
            return results
        
        class_folders = [f for f in os.listdir(categories_path) 
                        if os.path.isdir(os.path.join(categories_path, f))]
        
        logging.info(f"Found {len(class_folders)} class folders in dataset")
        
        start_time = time.time()
        
        # Create text features using baseline method
        text_features = self.create_text_features_baseline(CALTECH101_CLASSES, templates)
        
        matched_classes = 0
        for class_name in CALTECH101_CLASSES:
            matching_folder = self.find_matching_class_folder(class_folders, class_name)
            if not matching_folder:
                continue
            
            matched_classes += 1
            class_dir = os.path.join(categories_path, matching_folder)
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not image_files:
                continue
            
            image_files = image_files[:max_images_per_class]
            
            class_correct = 0
            class_total = 0
            
            for img_file in image_files:
                image_path = os.path.join(class_dir, img_file)
                
                predictions = self._predict_with_features_baseline(image_path, text_features)
                
                if predictions is not None and len(predictions) > 0:
                    top_prediction = predictions[0]
                    
                    is_correct = top_prediction['class'] == class_name
                    
                    prediction_record = {
                        'true_class': class_name,
                        'predicted_class': top_prediction['class'],
                        'confidence': top_prediction['confidence'],
                        'is_correct': is_correct
                    }
                    
                    results['predictions'].append(prediction_record)
                    
                    if is_correct:
                        results['correct_predictions'] += 1
                        class_correct += 1
                    
                    results['total_images'] += 1
                    class_total += 1
            
            if class_total > 0:
                class_acc = class_correct / class_total
                results['per_class_accuracy'][class_name] = class_acc
        
        if results['total_images'] > 0:
            results['accuracy'] = results['correct_predictions'] / results['total_images']
        results['inference_time'] = time.time() - start_time
        
        logging.info(f"Baseline {method_name} completed: Accuracy = {results['accuracy']:.4f}, "
                    f"Matched {matched_classes}/102 classes, Time = {results['inference_time']:.2f}s")
        
        return results

    def evaluate_ensemble_method(self, dataset_path, method_name, templates, max_images_per_class=20):
        """Evaluate ensemble methods with different implementation strategies"""
        logging.info(f"Evaluating ensemble method: {method_name}")
        
        results = {
            'method_name': method_name,
            'description': self._get_method_description(method_name),
            'correct_predictions': 0,
            'total_images': 0,
            'accuracy': 0.0,
            'inference_time': 0.0,
            'templates_used': len(templates),
            'per_class_accuracy': {},
            'predictions': []
        }
        
        categories_path = os.path.join(dataset_path, "101_ObjectCategories")
        
        if not os.path.exists(categories_path):
            logging.error(f"Dataset directory not found: {categories_path}")
            return results
        
        class_folders = [f for f in os.listdir(categories_path) 
                        if os.path.isdir(os.path.join(categories_path, f))]
        
        start_time = time.time()
        
        matched_classes = 0
        for class_name in CALTECH101_CLASSES:
            matching_folder = self.find_matching_class_folder(class_folders, class_name)
            
            if not matching_folder:
                continue
            
            matched_classes += 1
            class_dir = os.path.join(categories_path, matching_folder)
            image_files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if not image_files:
                continue
            
            image_files = image_files[:max_images_per_class]
            
            class_correct = 0
            class_total = 0
            
            for img_file in image_files:
                image_path = os.path.join(class_dir, img_file)
                
                if method_name == "feature_average":
                    text_features = self.method1_feature_average(CALTECH101_CLASSES, templates)
                    predictions = self._predict_with_features_baseline(image_path, text_features)
                elif method_name == "prediction_average":
                    predictions = self.method2_prediction_average(image_path, CALTECH101_CLASSES, templates)
                elif method_name == "majority_voting":
                    predictions = self.method3_majority_voting(image_path, CALTECH101_CLASSES, templates)
                elif method_name == "weighted_prediction":
                    predictions = self.method4_weighted_prediction(image_path, CALTECH101_CLASSES, templates)
                elif method_name == "learned_weighting":
                    predictions = self.method5_learned_weighting(image_path, CALTECH101_CLASSES, templates)
                
                if predictions is not None and len(predictions) > 0:
                    top_prediction = predictions[0]
                    
                    is_correct = top_prediction['class'] == class_name
                    
                    prediction_record = {
                        'true_class': class_name,
                        'predicted_class': top_prediction['class'],
                        'confidence': top_prediction['confidence'],
                        'is_correct': is_correct
                    }
                    
                    results['predictions'].append(prediction_record)
                    
                    if is_correct:
                        results['correct_predictions'] += 1
                        class_correct += 1
                    
                    results['total_images'] += 1
                    class_total += 1
            
            if class_total > 0:
                class_acc = class_correct / class_total
                results['per_class_accuracy'][class_name] = class_acc
        
        if results['total_images'] > 0:
            results['accuracy'] = results['correct_predictions'] / results['total_images']
        results['inference_time'] = time.time() - start_time
        
        logging.info(f"Ensemble method {method_name} completed: Accuracy = {results['accuracy']:.4f}, "
                    f"Matched {matched_classes}/102 classes, Time = {results['inference_time']:.2f}s")
        
        return results

    def _predict_with_features_baseline(self, image_path, text_features):
        """Helper method for prediction with precomputed features"""
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                if text_features.device != self.device:
                    text_features = text_features.to(self.device)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(5)
                
            predictions = []
            for value, idx in zip(values, indices):
                class_idx = idx.item()
                if class_idx < len(CALTECH101_CLASSES):
                    predictions.append({
                        'class': CALTECH101_CLASSES[class_idx],
                        'confidence': value.item()
                    })
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error in _predict_with_features_baseline for {image_path}: {str(e)}")
            return None

    def _get_method_description(self, method_name):
        """Get description for each ensemble method"""
        descriptions = {
            "feature_average": "Average text features across all templates before similarity computation",
            "prediction_average": "Average prediction scores across templates after individual predictions",
            "majority_voting": "Each template votes for one class, majority class wins",
            "weighted_prediction": "Weighted average of predictions based on template importance",
            "learned_weighting": "Logistic regression learned weights for templates",
            "selective_ensemble": "Select top-k templates based on individual performance",
            "random_forest_weighting": "Random Forest learned template weights",
            "variance_weighting": "Variance-based template weighting"
        }
        return descriptions.get(method_name, "Unknown method")

    def calculate_per_class_accuracy(self, true_labels, predictions, classes):
        """Calculate per-class accuracy"""
        class_correct = {cls: 0 for cls in classes}
        class_total = {cls: 0 for cls in classes}
        
        for true_label, pred_label in zip(true_labels, predictions):
            true_class = classes[true_label]
            pred_class = classes[pred_label]
            
            class_total[true_class] += 1
            if true_class == pred_class:
                class_correct[true_class] += 1
        
        class_accuracy = {}
        for cls in classes:
            if class_total[cls] > 0:
                class_accuracy[cls] = class_correct[cls] / class_total[cls]
            else:
                class_accuracy[cls] = 0.0
        
        return class_accuracy