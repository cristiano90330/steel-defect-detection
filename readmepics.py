from ultralytics import YOLO
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 旧基线（最开始那个好模型）
    model_old = YOLO(r"D:\ultralytics\runs\detect\steel_defect_model7\weights\best.pt")
    # 新最终版
    model_new = YOLO(r"D:\ultralytics\runs\detect\steel_final_version\weights\best.pt")

    data = r"D:\ultralytics\works\NEU-DET\data.yaml"

    res_old = model_old.val(data=data, workers=0, verbose=False)
    res_new = model_new.val(data=data, workers=0, verbose=False)

    names = list(res_old.names.values())
    ap_old = res_old.box.ap50
    ap_new = res_new.box.ap50

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10,5))
    plt.bar(names, ap_old, alpha=0.5, label='优化前')
    plt.bar(names, ap_new, alpha=0.8, label='最终稳定版')
    plt.xticks(rotation=20)
    plt.title('模型优化前后 AP 对比')
    plt.legend()
    plt.tight_layout()
    plt.savefig('final_ap_compare.png')
    print("已保存最终对比图：final_ap_compare.png")