import torch
import cv2
import numpy as np
import os
import time
import warnings
import sys
from ultralytics import YOLO

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("Starting program...")

# カレントディレクトリの取得
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# PyTorchのバージョンとCUDAの状態を表示
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU for processing")
    device = torch.device("cuda")
else:
    print("Using CPU for processing")
    device = torch.device("cpu")

# YOLOv8セグメンテーションモデルの読み込み
print("Loading YOLOv8 segmentation model...")
model_path =  r"E:\Programing\Python\0_Yolo\yolov8m-seg.pt"  # より大きなモデルを使用
if os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
else:
    print("Model file not found. Downloading from the internet...")
    model = YOLO('yolov8m-seg.pt')  # より大きなモデルを使用
print("YOLOv8 segmentation model loaded successfully")

# 画像の読み込み
image_path = r"E:\Programing\20250405_object_recognition\Picuture\WIN_20250405_10_38_44_Pro.jpg"
print(f"Loading image from: {image_path}")
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read image at {image_path}")
    exit()
print(f"Image loaded successfully. Shape: {image.shape}")

# 画像のサイズを小さくする（処理を高速化するため）
max_size = 800
height, width = image.shape[:2]
if height > max_size or width > max_size:
    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = cv2.resize(image, (new_width, new_height))
    print(f"Image resized to: {image.shape}")

# 処理時間の計測開始
start_time = time.time()
print("Starting image processing...")

# 人物検出とセグメンテーション
print("Detecting people and performing segmentation...")
results = model(image)
print("People detection and segmentation completed")

# 検出された人物を黒塗り
if len(results) > 0:
    print("People detected. Applying blackout...")
    # 人物のマスクを取得
    masks = results[0].masks
    
    if masks is not None:
        for i, mask in enumerate(masks):
            # クラスIDを取得
            class_id = results[0].boxes[i].cls.item()
            
            # 人物クラス（クラスID: 0）のみを対象とする
            if class_id == 0:
                # マスクを二値化
                binary_mask = mask.data[0].cpu().numpy().astype(np.uint8) * 255
                
                # マスクをリサイズ（リサイズ後の画像のサイズに合わせる）
                binary_mask = cv2.resize(binary_mask, (new_width, new_height))
                
                # マスクを膨張（膨張の度合いを大きくする）
                kernel = np.ones((7, 7), np.uint8)
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
                
                # マスクを平滑化（平滑化の度合いを大きくする）
                binary_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)
                
                # マスクを使用して黒塗り（閾値を下げる）
                image[binary_mask > 64] = 0
        
        print("Blackout applied successfully")
    else:
        print("No people detected in the image")
else:
    print("No people detected in the image")

# 処理時間の計測終了
end_time = time.time()
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.4f} seconds")

# 結果の保存
output_path = r"E:\Programing\20250405_object_recognition\Picuture\output_pytorch_yolo8_3.jpg"
print(f"Saving result to: {output_path}")
cv2.imwrite(output_path, image)
print(f"Processed image saved to {output_path}")

print("Program completed successfully")