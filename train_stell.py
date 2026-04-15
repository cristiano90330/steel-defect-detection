from ultralytics import YOLO

# 主程序入口（Windows必须加）
if __name__ == '__main__':
    # ----------------------- 1. 加载基础模型 -----------------------
    # 从你最好的模型开始（不是从头训）→ 保证精度不丢
    model = YOLO(r"D:\ultralytics\runs\detect\steel_defect_model7\weights\best.pt")

    # ----------------------- 2. 训练配置（全解释版） -----------------------
    model.train(
        # 数据集配置
        data=r"D:\ultralytics\works\NEU-DET\data.yaml",  # 数据集路径

        # 训练轮次：短、精、准，不会过拟合
        epochs=30,

        # 图片尺寸：640是平衡精度/速度最安全值
        imgsz=640,

        # 批次：4060笔记本最稳的值
        batch=4,

        # 使用GPU
        device=0,

        # 保存文件夹名称
        name="steel_final_version",

        # Windows系统必须加，防止报错
        workers=0,
        # resume=True,  # 强制续训
        amp=False,  # 关闭网络检查

        # ===================== 【核心：安全增强】 =====================
        # 只开【不会破坏缺陷纹理】的增强，绝对稳定
        fliplr=0.5,  # 左右翻转：安全，所有缺陷通用
        flipud=0.2,  # 上下翻转：轻度，不破坏结构
        degrees=5,  # 旋转±5°：极轻度，不破坏裂纹
        hsv_h=0.01,  # 色调：几乎不变
        hsv_s=0.4,  # 饱和度：轻微变化
        hsv_v=0.2,  # 亮度：极轻微变化

        mosaic=0.5,  # 马赛克：半开，不破坏细缺陷
        mixup=0.0,  # 关闭！mixup会模糊缺陷 → 必关！

        # ===================== 【训练策略】 =====================
        patience=5,  # 早停：5轮不涨自动停，防止过拟合
        save=True,  # 保存最优模型
    )