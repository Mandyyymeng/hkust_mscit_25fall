import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import clip
from config import SIMPLE_TEMPLATES, ENSEMBLE_TEMPLATES, CALTECH101_CLASSES

class CLIPResultsVisualizer:
    def __init__(self, results_dir="experiment_results", model_name="ViT-B/32", dataset_path="./caltech-101"):
        self.results_dir = results_dir
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.all_results = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Set matplotlib style to match the example exactly
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['font.size'] = 12
        
    def load_results(self):
        """Load experiment results"""
        json_path = os.path.join(self.results_dir, 'detailed_results.json')
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Results file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            self.all_results = json.load(f)
            
        print(f"Loaded results for {len(self.all_results)} methods")
        return self.all_results
    
    def load_sample_images_and_texts(self, num_classes=8, num_images_per_class=1):
        """Load sample images and create text descriptions"""
        categories_dir = os.path.join(self.dataset_path, "101_ObjectCategories")
        
        if not os.path.exists(categories_dir):
            print(f"Dataset directory not found: {categories_dir}")
            return None, None, None
        
        # Get available categories
        available_categories = [d for d in os.listdir(categories_dir) 
                              if os.path.isdir(os.path.join(categories_dir, d))]
        
        # Use first num_classes categories
        selected_categories = available_categories[:num_classes]
        
        original_images = []
        processed_images = []
        texts = []
        descriptions = {}
        
        # Create text descriptions using simple templates
        template = SIMPLE_TEMPLATES[0]  # Use the first simple template
        
        plt.figure(figsize=(16, 8))
        
        image_count = 0
        for i, category in enumerate(selected_categories):
            category_path = os.path.join(categories_dir, category)
            
            if not os.path.exists(category_path):
                continue
                
            # Get images in category
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                continue
                
            # Take first image
            image_path = os.path.join(category_path, image_files[0])
            
            try:
                # Load and process image
                image = Image.open(image_path).convert("RGB")
                
                # Display in subplot
                plt.subplot(2, 4, i + 1)
                plt.imshow(image)
                plt.title(f"{category}\n{template.format(category)}", fontsize=10)
                plt.xticks([])
                plt.yticks([])
                
                original_images.append(image)
                processed_images.append(self.preprocess(image))
                
                # Create text description
                text_description = template.format(category)
                texts.append(text_description)
                descriptions[category] = text_description
                
                image_count += 1
                
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
        
        plt.tight_layout()
        plt.show()
        
        return original_images, processed_images, texts, descriptions
    
    def create_similarity_matrix_visualization(self, method_name="simple_baseline", save_path=None):
        """Create exact CLIP tutorial style similarity matrix visualization"""
        print("Loading sample images and creating text descriptions...")
        
        # Load sample images and texts
        original_images, processed_images, texts, descriptions = self.load_sample_images_and_texts()
        
        if not original_images:
            print("No images found. Creating placeholder visualization...")
            self._create_placeholder_similarity_matrix()
            return
        
        print("Computing image and text features...")
        
        # Prepare images and texts for CLIP
        image_input = torch.tensor(np.stack(processed_images)).to(self.device)
        text_tokens = clip.tokenize(texts).to(self.device)
        
        # Compute features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity matrix
        similarity = (text_features @ image_features.T).cpu().numpy()
        
        count = len(descriptions)
        
        # Create the exact visualization from CLIP tutorial
        plt.figure(figsize=(15, 10))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)
        
        # Remove colorbar as in the example
        # plt.colorbar()
        
        plt.yticks(range(count), texts, fontsize=14)
        plt.xticks([])
        
        # Add images below the matrix
        for i, image in enumerate(original_images):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
        
        # Add similarity values as text
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.2f}", 
                        ha="center", va="center", size=10)
        
        # Remove spines
        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)
        
        plt.xlim([-0.5, count - 0.5])
        plt.ylim([count + 0.5, -2])
        
        plt.title(f"Cosine similarity between text and image features\nMethod: {method_name}", size=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Similarity matrix saved to: {save_path}")
        
        plt.show()
        
        return similarity
    
    def _create_placeholder_similarity_matrix(self):
        """Create a placeholder visualization when images are not available"""
        # Sample classes and descriptions
        sample_classes = CALTECH101_CLASSES[:8]
        template = SIMPLE_TEMPLATES[0]
        
        texts = [template.format(cls) for cls in sample_classes]
        descriptions = {cls: template.format(cls) for cls in sample_classes}
        
        # Create synthetic similarity matrix
        count = len(sample_classes)
        similarity = np.random.uniform(0.1, 0.3, (count, count))
        np.fill_diagonal(similarity, np.random.uniform(0.25, 0.35, count))
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)
        
        plt.yticks(range(count), texts, fontsize=14)
        plt.xticks([])
        
        # Add placeholder images (colored squares)
        for i in range(count):
            color = np.random.rand(3,)
            placeholder = np.ones((100, 100, 3)) * color.reshape(1, 1, 3)
            plt.imshow(placeholder, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
        
        # Add similarity values
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.2f}", 
                        ha="center", va="center", size=10)
        
        # Remove spines
        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)
        
        plt.xlim([-0.5, count - 0.5])
        plt.ylim([count + 0.5, -2])
        
        plt.title("Cosine similarity between text and image features (Placeholder)", size=16)
        plt.tight_layout()
        plt.show()

    def visualize_simple_templates(self, save_path="simple_templates_visualization.png"):
        """Visualize simple templates with clean layout"""
        original_images, processed_images, texts, descriptions = self.load_sample_images_and_texts(num_classes=4)
        if not original_images:
            print("No images found.")
            return
        
        sample_classes = list(descriptions.keys())[:4]
        simple_templates = SIMPLE_TEMPLATES[:3]
        
        # Create clean layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Template display - clean version
        ax1.axis('off')
        
        # Display templates with classes
        template_content = "Simple Templates:\n\n"
        for i, template in enumerate(simple_templates):
            template_content += f"Template {i+1}: {template}\n"
            for j, cls in enumerate(sample_classes):
                full_text = template.format(cls)
                template_content += f"   • {full_text}\n"
            template_content += "\n"
        
        ax1.text(0.02, 0.98, template_content, transform=ax1.transAxes, 
                fontsize=11, family='monospace', 
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.3))
        
        # Image display - fixed orientation
        ax2.axis('off')
        
        # Create image grid
        for i, (cls, img) in enumerate(zip(sample_classes, original_images)):
            # Ensure image is properly oriented
            img_array = np.array(img)
            
            # Display image in correct orientation
            ax_img = fig.add_axes([0.1 + i*0.2, 0.1, 0.15, 0.15])
            ax_img.imshow(img_array)
            ax_img.axis('off')
            ax_img.set_title(cls, fontsize=10, pad=5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.show()

    def visualize_ensemble_templates(self, save_path="ensemble_templates_visualization.png"):
        """Visualize ensemble templates with clean layout"""
        original_images, processed_images, texts, descriptions = self.load_sample_images_and_texts(num_classes=4)
        if not original_images:
            print("No images found.")
            return
        
        sample_classes = list(descriptions.keys())[:4]
        ensemble_templates = ENSEMBLE_TEMPLATES[:6]  # Show more ensemble templates
        
        # Create clean layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Template display - clean version
        ax1.axis('off')
        
        # Display templates with classes
        template_content = "Ensemble Templates:\n\n"
        for i, template in enumerate(ensemble_templates):
            template_content += f"Template {i+1}: {template}\n"
            for j, cls in enumerate(sample_classes):
                full_text = template.format(cls)
                template_content += f"   • {full_text}\n"
            template_content += "\n"
        
        ax1.text(0.02, 0.98, template_content, transform=ax1.transAxes, 
                fontsize=10, family='monospace', 
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.3))
        
        # Image display - fixed orientation
        ax2.axis('off')
        
        # Create image grid
        for i, (cls, img) in enumerate(zip(sample_classes, original_images)):
            # Ensure image is properly oriented
            img_array = np.array(img)
            
            # Display image in correct orientation
            ax_img = fig.add_axes([0.1 + i*0.2, 0.1, 0.15, 0.15])
            ax_img.imshow(img_array)
            ax_img.axis('off')
            ax_img.set_title(cls, fontsize=10, pad=5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.show()
    
def main():
    """Main function to run visualizations"""
    visualizer = CLIPResultsVisualizer(dataset_path="./caltech-101")
    
    try:
        # Generate similarity matrix visualization
        print("Creating CLIP-style similarity matrix visualization...")
        visualizer.create_similarity_matrix_visualization("simple_baseline", 
                                                         "clip_similarity_matrix.png")
        
        # 独立显示简单模板（单独窗口）
        print("\nCreating simple templates visualization...")
        visualizer.visualize_simple_templates()
        
        # 独立显示集成模板（单独窗口）
        print("\nCreating ensemble templates visualization...")
        visualizer.visualize_ensemble_templates()
        
        print("\nVisualization completed successfully!")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()