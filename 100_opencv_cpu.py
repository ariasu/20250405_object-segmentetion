import cv2
import mediapipe as mp
import numpy as np
import os
import time
import warnings

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("Starting program...")

# カレントディレクトリの取得
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# OpenCVのバージョンとCUDAの状態を表示
print("OpenCV version:", cv2.__version__)
print("CUDA available:", cv2.cuda.getCudaEnabledDeviceCount() > 0)
print("Using CPU for processing")

# MediaPipeの初期化
print("Initializing MediaPipe...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,  # モデルの複雑さを下げる（0: 軽量, 1: フル, 2: 重い）
    min_detection_confidence=0.5,
    enable_segmentation=True  # セグメンテーションを有効化
)
print("MediaPipe initialized successfully")

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

# BGRからRGBに変換（MediaPipeはRGB形式を要求）
print("Converting BGR to RGB...")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("BGR to RGB conversion completed")

# 人物検出
print("Detecting people...")
results = pose.process(image_rgb)
print("People detection completed")

# 検出された人物を黒塗り
if results.pose_landmarks:
    print("People detected. Applying blackout...")
    # 人物の境界ボックスを計算
    h, w = image.shape[:2]
    landmarks = results.pose_landmarks.landmark
    
    # すべてのランドマークの座標を取得
    x_coordinates = [landmark.x for landmark in landmarks]
    y_coordinates = [landmark.y for landmark in landmarks]
    
    # 境界ボックスの座標を計算
    x_min = int(min(x_coordinates) * w)
    x_max = int(max(x_coordinates) * w)
    y_min = int(min(y_coordinates) * h)
    y_max = int(max(y_coordinates) * h)
    
    # 余白を追加
    padding = 20
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    
    # CPUで黒塗り処理
    image[y_min:y_max, x_min:x_max] = 0
    print("Blackout applied successfully")
else:
    print("No people detected in the image")

# 処理時間の計測終了
end_time = time.time()
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.4f} seconds")

# 結果の保存
output_path = r"E:\Programing\20250405_object_recognition\Picuture\output_opencv_cpu.jpg"
print(f"Saving result to: {output_path}")
cv2.imwrite(output_path, image)
print(f"Processed image saved to {output_path}")

print("Program completed successfully")