from flask import Flask, request, render_template_string, send_from_directory
from ultralytics import YOLO
import os
import cv2
import uuid
import logging
from html import escape
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "steel_improved_v14", "weights", "best.pt")

model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


INDEX_HTML = '''<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>钢铁缺陷检测系统</title>
    <style>
        body { font-family: "Microsoft YaHei", Arial, sans-serif; text-align: center;
               margin-top: 50px; background: #f5f7fa; }
        h1 { color: #2c3e50; }
        .upload-box { margin: 30px auto; width: 500px; padding: 20px;
                      background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        button { padding: 10px 24px; background: #3498db; color: white; border: none;
                 border-radius: 4px; cursor: pointer; font-size: 14px; }
        button:hover { background: #2980b9; }
    </style>
</head>
<body>
    <h1>钢铁表面缺陷检测系统</h1>
    <div class="upload-box">
        <form action="/detect" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br><br>
            <button type="submit">开始检测</button>
        </form>
    </div>
</body>
</html>'''

RESULT_HTML = '''<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>检测结果</title>
    <style>
        body { font-family: "Microsoft YaHei", Arial, sans-serif; text-align: center;
               margin-top: 30px; background: #f5f7fa; }
        h1 { color: #2c3e50; }
        .result-box { margin: 20px auto; width: 800px; padding: 20px;
                      background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        img { max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; }
        a { color: #3498db; text-decoration: none; }
        .defect-item { padding: 6px 12px; margin: 4px; display: inline-block;
                       background: #eef2f7; border-radius: 4px; }
        .low-conf { color: #e74c3c; }
        .high-conf { color: #27ae60; }
    </style>
</head>
<body>
    <h1>检测完成</h1>
    <div class="result-box">
        <h3>识别结果：</h3>
        {% if defects %}
            {% for d in defects %}
                <span class="defect-item {{ 'high-conf' if d['置信度'] >= 0.7 else 'low-conf' }}">
                    {{ d['类别'] }} (置信度: {{ "%.2f"|format(d['置信度']) }})
                </span>
            {% endfor %}
        {% else %}
            <p>未检测到缺陷</p>
        {% endif %}
    </div>
    <div class="result-box">
        <h3>检测图片：</h3>
        <img src="/uploads/{{ result_filename }}" alt="检测结果">
    </div>
    <br>
    <a href="/">返回重新检测</a>
</body>
</html>'''


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return "未上传文件", 400

    file = request.files['file']
    if not file or file.filename == '':
        return "未选择文件", 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return "仅支持 JPG/PNG/BMP 格式", 400

    safe_basename = f"{uuid.uuid4().hex}_{filename}"
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_basename)
    file.save(img_path)

    try:
        results = model.predict(img_path, conf=0.25, verbose=False)
        result = results[0]

        img_with_boxes = result.plot()
        result_basename = f"result_{safe_basename}"
        result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], result_basename)
        cv2.imwrite(result_img_path, img_with_boxes)

        defects = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_name = result.names[int(box.cls[0])]
                conf = round(float(box.conf[0]), 3)
                defects.append({"类别": escape(cls_name), "置信度": conf})

        logging.info(f"检测完成: {filename}, 检测到 {len(defects)} 个缺陷")
    except Exception as e:
        logging.error(f"推理失败: {e}")
        return "检测过程出错，请重试", 500

    return render_template_string(RESULT_HTML,
                                  defects=defects,
                                  result_filename=escape(result_basename, quote=False))


@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    safe_name = secure_filename(os.path.basename(filename))
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
