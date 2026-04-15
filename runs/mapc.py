import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

# 你的文件路径
df = pd.read_csv(r"D:\ultralytics\runs\detect\steel_defect_model_s_lr_001\results.csv")

# 【关键】把所有列名的空格全部删掉！
df.columns = [col.strip() for col in df.columns]


# 画图
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50(B)', linewidth=2)
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95(B)', linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("YOLOv8s 小学习率 mAP 曲线")
plt.legend()
plt.grid(True)
plt.savefig("mAP_lr_001.png", dpi=300, bbox_inches='tight')
plt.show()