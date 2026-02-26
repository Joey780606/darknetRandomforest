"""
generate_train_val.py

掃描圖片目錄，依 80/20 比例隨機分割為訓練集和驗證集，
產生 train.txt 和 val.txt 路徑清單檔案。

使用方式：
    python scripts/generate_train_val.py

可選參數：
    --images_dir   圖片目錄（預設: data/images/train）
    --labels_dir   標註目錄（預設: data/labels/train）
    --output_dir   輸出 train.txt / val.txt 的目錄（預設: data）
    --split_ratio  訓練集比例（預設: 0.8）
    --seed         隨機種子（預設: 42）
"""

import argparse
import os
import random
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(images_dir):
    """掃描目錄中的所有圖片檔案。"""
    images = []
    for f in os.listdir(images_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(f)
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description="產生 train.txt 和 val.txt 路徑清單")
    parser.add_argument("--images_dir", default="data/images/train", help="圖片來源目錄")
    parser.add_argument("--labels_dir", default="data/labels/train", help="標註來源目錄")
    parser.add_argument("--output_dir", default="data", help="輸出目錄")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="訓練集比例 (0~1)")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    args = parser.parse_args()

    # 找到所有圖片
    images = find_images(args.images_dir)
    print(f"找到 {len(images)} 張圖片")

    if len(images) == 0:
        print("錯誤：圖片目錄中沒有圖片！")
        print(f"  請先執行 convert_300w_to_yolo.py 將資料集放入 {args.images_dir}")
        return

    # 檢查每張圖片是否有對應的標註檔
    valid_images = []
    for img_name in images:
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(args.labels_dir, label_name)
        if os.path.exists(label_path):
            valid_images.append(img_name)
        else:
            print(f"  警告：{img_name} 沒有對應的標註檔，已跳過")

    print(f"有效圖片（含標註）: {len(valid_images)} 張")

    # 隨機分割
    random.seed(args.seed)
    random.shuffle(valid_images)

    split_idx = int(len(valid_images) * args.split_ratio)
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]

    print(f"訓練集: {len(train_images)} 張")
    print(f"驗證集: {len(val_images)} 張")

    # 建立驗證集目錄
    val_images_dir = os.path.join(os.path.dirname(args.images_dir), "val")
    val_labels_dir = os.path.join(os.path.dirname(args.labels_dir), "val")
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 移動驗證集的圖片和標註到 val 目錄
    for img_name in val_images:
        label_name = os.path.splitext(img_name)[0] + ".txt"

        src_img = os.path.join(args.images_dir, img_name)
        src_label = os.path.join(args.labels_dir, label_name)
        dst_img = os.path.join(val_images_dir, img_name)
        dst_label = os.path.join(val_labels_dir, label_name)

        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

    # 產生 train.txt
    os.makedirs(args.output_dir, exist_ok=True)
    train_txt = os.path.join(args.output_dir, "train.txt")
    with open(train_txt, "w") as f:
        for img_name in train_images:
            img_path = os.path.join(args.images_dir, img_name)
            f.write(img_path + "\n")

    # 產生 val.txt
    val_txt = os.path.join(args.output_dir, "val.txt")
    with open(val_txt, "w") as f:
        for img_name in val_images:
            img_path = os.path.join(val_images_dir, img_name)
            f.write(img_path + "\n")

    print(f"\n路徑清單已產生：")
    print(f"  {train_txt} ({len(train_images)} 張)")
    print(f"  {val_txt} ({len(val_images)} 張)")
    print(f"\n驗證集已移動到：")
    print(f"  圖片: {val_images_dir}")
    print(f"  標註: {val_labels_dir}")


if __name__ == "__main__":
    main()
