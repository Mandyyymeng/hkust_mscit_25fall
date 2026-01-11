from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as TF_MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50 as TF_ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions as mobilenet_decode
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.resnet50 import decode_predictions as resnet_decode
from tensorflow.keras.preprocessing import image
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 加载两个模型
mobilenet_model = TF_MobileNetV2(weights='imagenet')
resnet_model = TF_ResNet50(weights='imagenet')

image_files = [f for f in os.listdir('images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

with open('model_comparison.txt', 'w') as f:
    for img_file in image_files:
        img_path = os.path.join('images', img_file)

        f.write(f"Image: {img_file}\n")
        f.write("=" * 50 + "\n")

        # MobileNetV2预测
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # MobileNetV2预处理和预测
        x_mobilenet = mobilenet_preprocess(x.copy())
        preds_mobilenet = mobilenet_model.predict(x_mobilenet, verbose=0)
        results_mobilenet = mobilenet_decode(preds_mobilenet, top=3)[0]

        f.write("MobileNetV2:\n")
        for i, (_, label, score) in enumerate(results_mobilenet):
            f.write(f"  {i + 1}. {label} ({score:.4f})\n")

        # ResNet50预处理和预测
        x_resnet = resnet_preprocess(x.copy())
        preds_resnet = resnet_model.predict(x_resnet, verbose=0)
        results_resnet = resnet_decode(preds_resnet, top=3)[0]

        f.write("ResNet50:\n")
        for i, (_, label, score) in enumerate(results_resnet):
            f.write(f"  {i + 1}. {label} ({score:.4f})\n")

        f.write("\n" + "=" * 50 + "\n\n")

print("Model comparison saved to model_comparison.txt")