from ultralytics import YOLO
import os
import cv2

# 1. 加载你最优的基准s模型（路径完全对应你的文件夹）
model = YOLO(r"D:\ultralytics\runs\detect\steel_defect_model7\weights\best.pt")

# 2. 验证集图片路径（对应你的NEU-DET数据集）
test_img_dir = r"D:\ultralytics\works\NEU-DET\valid\images"
# 3. 结果保存路径（自动创建文件夹）
save_dir = r"D:\ultralytics\works\NEU-DET\inference_results"
os.makedirs(save_dir, exist_ok=True)

# 4. 批量推理
for img_name in os.listdir(test_img_dir):
    # 只处理图片文件
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    img_path = os.path.join(test_img_dir, img_name)

    # 推理（conf=0.25置信度阈值，可根据需要调整）
    results = model(img_path, conf=0.25, save=False)

    # 绘制检测框并保存
    for result in results:
        # 自动绘制框、标签、置信度，和你训练时生成的val_batch0_pred.jpg效果一致
        img = result.plot()
        # 保存到结果文件夹
        cv2.imwrite(os.path.join(save_dir, img_name), img)

print(f"✅ 批量推理完成！所有带检测框的图片已保存到：{save_dir}")