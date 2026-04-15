from flask import Flask, request, render_template_string, send_from_directory
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename

# 初始化 Flask
app = Flask(__name__)

# 1. 加载模型（绝对路径锁死，不会错）
model = YOLO(r"D:\ultralytics\runs\detect\steel_defect_model7\weights\best.pt")  # 换回.pt，推理更稳，置信度更准

# 2. 上传目录（绝对路径，避免相对路径问题）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 3. 首页：上传界面
@app.route('/')
def index():
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>钢铁缺陷检测系统</title>
        <style>
            body { font-family: "Microsoft YaHei", Arial; text-align: center; margin-top: 50px; }
            h1 { color: #2c3e50; }
            .upload-box { margin: 30px auto; width: 500px; padding: 20px; border: 1px solid #eee; border-radius: 8px; }
            button { padding: 8px 20px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; }
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
    </html>
    ''')

# 4. 检测接口
@app.route('/detect', methods=['POST'])
def detect():
    # 获取上传文件
    file = request.files['file']
    if not file or file.filename == '':
        return "未上传文件", 400

    # 安全保存原图
    filename = secure_filename(file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_path)

    # 模型推理（conf=0.25，过滤低置信度框）
    results = model.predict(img_path, conf=0.25, verbose=False)
    result = results[0]

    # 绘制带框图片
    img_with_boxes = result.plot()
    result_filename = f"result_{filename}"
    result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_img_path, img_with_boxes)

    # 解析检测结果
    defects = []
    for box in result.boxes:
        cls_name = result.names[int(box.cls[0])]
        conf = round(float(box.conf[0]), 3)
        defects.append({"类别": cls_name, "置信度": conf})

    # 返回结果页（图片直接显示）
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>检测结果</title>
        <style>
            body { font-family: "Microsoft YaHei", Arial; text-align: center; margin-top: 30px; }
            h1 { color: #2c3e50; }
            .result-box { margin: 20px auto; width: 800px; padding: 20px; border: 1px solid #eee; border-radius: 8px; }
            img { max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; }
            a { color: #3498db; text-decoration: none; }
        </style>
    </head>
    <body>
        <h1>检测完成</h1>
        <div class="result-box">
            <h3>识别结果：</h3>
            {% for d in defects %}
                <p>{{ d.类别 }} (置信度: {{ d.置信度 }})</p>
            {% endfor %}
        </div>
        <div class="result-box">
            <h3>检测图片：</h3>
            <img src="/uploads/{{ result_filename }}" alt="检测结果">
        </div>
        <br>
        <a href="/">返回重新检测</a>
    </body>
    </html>
    ''', defects=defects, result_filename=result_filename)

# 5. 静态文件路由（专门解决图片不显示！）
@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 启动服务
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)  # 关闭debug，更稳定