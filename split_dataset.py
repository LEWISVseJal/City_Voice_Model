import os
import shutil
import random

source = "dataset/train"
output_train = "dataset_new/train"
output_val = "dataset_new/val"

split_ratio = 0.8  # 80% train, 20% val

for class_name in os.listdir(source):
    class_path = os.path.join(source, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split = int(len(images) * split_ratio)
    train_images = images[:split]
    val_images = images[split:]

    os.makedirs(os.path.join(output_train, class_name), exist_ok=True)
    os.makedirs(os.path.join(output_val, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(output_train, class_name, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(output_val, class_name, img)
        )

    print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")

print("\nDone! Use dataset_new/ for training.")