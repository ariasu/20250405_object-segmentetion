from flask import Flask, request, render_template, send_file
import os
from ultralytics import YOLO
import cv2
import numpy as np
import warnings

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

app = Flask(__name__)

# アップロードされたファイルを保存するディレクトリ
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            output_path = process_image(filename)
            return send_file(output_path, as_attachment=True)
    return render_template('upload.html')

def process_image(image_path):
    # YOLOv8モデルの読み込み
    model = YOLO('yolov8m-seg.pt')
    
    # 画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # 人物検出の実行
    results = model(image)
    
    # 検出結果の処理
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # 人物クラス
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 人物部分を黒塗り
                image[y1:y2, x1:x2] = 0
    
    # ダウンロードフォルダのパスを取得
    download_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
    output_path = os.path.join(download_folder, 'output.jpg')

    # 処理結果の保存（常に同じファイル名を使用し、上書きする）
    cv2.imwrite(output_path, image)

    print(f"Output path: {output_path}")

    # 処理結果のパスを返す
    return output_path

if __name__ == '__main__':
    app.run(debug=True)