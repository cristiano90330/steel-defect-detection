"""
TTA (Test-Time Augmentation) 推理脚本
对输入图片进行多角度推理后融合结果，特别提升裂纹、氧化皮等难检测类别的精度
"""
from ultralytics import YOLO
import cv2
import numpy as np
import os


def tta_predict(model, img_path, conf=0.25, iou=0.5):
    """对单张图片进行TTA推理：原图 + 水平翻转 + 垂直翻转 + 轻微旋转"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    variants = {
        'original': img,
        'hflip': cv2.flip(img, 1),
        'vflip': cv2.flip(img, 0),
    }
    # 轻微旋转版本（±2°不破坏细裂纹结构）
    variants['rot_p2'] = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 2, 1.0), (w, h))
    variants['rot_m2'] = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, -2, 1.0), (w, h))

    all_boxes = []
    for name, variant in variants.items():
        results = model(variant, conf=conf, verbose=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            # 将增强版本的框映射回原图坐标
            if name == 'hflip':
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            elif name == 'vflip':
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            elif name == 'rot_p2':
                rot_inv = cv2.getRotationMatrix2D(center, -2, 1.0)
                for j in range(len(boxes)):
                    corners = np.array([[boxes[j, 0], boxes[j, 1]], [boxes[j, 2], boxes[j, 1]],
                                       [boxes[j, 2], boxes[j, 3]], [boxes[j, 0], boxes[j, 3]]])
                    t = cv2.transform(corners.reshape(1, -1, 2), rot_inv).reshape(-1, 2)
                    boxes[j, 0] = max(0, t[:, 0].min()); boxes[j, 1] = max(0, t[:, 1].min())
                    boxes[j, 2] = min(w, t[:, 0].max()); boxes[j, 3] = min(h, t[:, 1].max())
            elif name == 'rot_m2':
                rot_inv = cv2.getRotationMatrix2D(center, 2, 1.0)
                for j in range(len(boxes)):
                    corners = np.array([[boxes[j, 0], boxes[j, 1]], [boxes[j, 2], boxes[j, 1]],
                                       [boxes[j, 2], boxes[j, 3]], [boxes[j, 0], boxes[j, 3]]])
                    t = cv2.transform(corners.reshape(1, -1, 2), rot_inv).reshape(-1, 2)
                    boxes[j, 0] = max(0, t[:, 0].min()); boxes[j, 1] = max(0, t[:, 1].min())
                    boxes[j, 2] = min(w, t[:, 0].max()); boxes[j, 3] = min(h, t[:, 1].max())

            for j in range(len(boxes)):
                all_boxes.append({'box': boxes[j].tolist(), 'conf': float(confs[j]), 'cls': int(cls_ids[j])})

    return all_boxes, img


def nms_tta(boxes_data, iou_thresh=0.5):
    """对TTA合并后的所有框做NMS去重，不同类别的框不会互相抑制"""
    if not boxes_data:
        return []
    boxes = np.array([b['box'] for b in boxes_data])
    scores = np.array([b['conf'] for b in boxes_data])
    cls_ids = np.array([b['cls'] for b in boxes_data])

    order = scores.argsort()[::-1]
    boxes = boxes[order]; scores = scores[order]; cls_ids = cls_ids[order]

    keep = []
    suppressed = set()
    for i in range(len(boxes)):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i + 1, len(boxes)):
            if j in suppressed:
                continue
            if cls_ids[i] != cls_ids[j]:
                continue
            x1 = max(boxes[i, 0], boxes[j, 0]); y1 = max(boxes[i, 1], boxes[j, 1])
            x2 = min(boxes[i, 2], boxes[j, 2]); y2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            iou_val = inter / (area_i + area_j - inter + 1e-6)
            if iou_val > iou_thresh:
                suppressed.add(j)

    return [{'box': boxes[i].tolist(), 'conf': float(scores[i]), 'cls': int(cls_ids[i])} for i in keep]


def draw_results(img, boxes_data, class_names):
    """绘制检测框"""
    img = img.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for b in boxes_data:
        x1, y1, x2, y2 = [int(v) for v in b['box']]
        color = colors[b['cls'] % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[b['cls']]} {b['conf']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


if __name__ == '__main__':
    model = YOLO(r"D:\ultralytics\steel-defect-detection\runs\detect\steel_improved_v14\weights\best.pt")
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

    test_img_dir = r"D:\ultralytics\steel-defect-detection\data\NEU-DET\split_val\images"
    save_dir = r"D:\ultralytics\steel-defect-detection\data\NEU-DET\tta_results"
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(test_img_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(test_img_dir, img_name)
        all_boxes, img = tta_predict(model, img_path, conf=0.2)
        final_boxes = nms_tta(all_boxes, iou_thresh=0.5)
        result_img = draw_results(img, final_boxes, class_names)
        cv2.imwrite(os.path.join(save_dir, img_name), result_img)
        defects = [f"{class_names[b['cls']]}:{b['conf']:.2f}" for b in final_boxes]
        print(f"{img_name}: {defects if defects else '无缺陷检出'}")

    print(f"\nTTA推理完成，结果保存至: {save_dir}")
