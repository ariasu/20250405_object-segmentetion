import torch
import cv2
import numpy as np
import os
import time
import warnings
import sys

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

# YOLOv5モデルの読み込み
print("Loading YOLOv5 model...")
model_path =  r"E:\Programing\Python\0_Yolo\yolov5s.pt"
if os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
else:
    print("Model file not found. Downloading from the internet...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)
print("YOLOv5 model loaded successfully")

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

# 人物検出
print("Detecting people...")
results = model(image)
print("People detection completed")

# 検出された人物を黒塗り
if len(results.xyxy[0]) > 0:
    print("People detected. Applying blackout...")
    # 人物の境界ボックスを取得
    boxes = results.xyxy[0].cpu().numpy()
    
    # 人物クラス（クラスID: 0）のボックスをフィルタリング
    person_boxes = boxes[boxes[:, 5] == 0]  # クラスID 0は人物
    
    if len(person_boxes) > 0:
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # 余白を追加
            padding = 20
            x1 = max(0, x1 - padding)
            x2 = min(new_width, x2 + padding)
            y1 = max(0, y1 - padding)
            y2 = min(new_height, y2 + padding)
            
            # 黒塗り処理
            image[y1:y2, x1:x2] = 0
        
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
output_path = r"E:\Programing\20250405_object_recognition\Picuture\output_pytorch_yolo5.jpg"
print(f"Saving result to: {output_path}")
cv2.imwrite(output_path, image)
print(f"Processed image saved to {output_path}")

print("Program completed successfully") 