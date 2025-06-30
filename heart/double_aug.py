import os
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from tqdm import tqdm

# ✅ 已划分好的训练路径
train_dir = "bispec_output/train"
out_dir_0 = os.path.join(train_dir, "0")
out_dir_1 = os.path.join(train_dir, "1")

# ✅ 设置增强策略
augment_funcs = [
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomHorizontalFlip(p=1.0)
]

def save_augmented(img_path, save_dir, label, num_augments):
    """读取图像并保存指定次数的增强版本"""
    img = Image.open(img_path).convert("RGB")
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    for i in range(num_augments):
        transform = augment_funcs[i % len(augment_funcs)]
        augmented = transform(img)
        new_name = f"{base_name}_aug{i}.png"
        augmented.save(os.path.join(save_dir, new_name))

# ✅ 执行增强
for label in ['0', '1']:
    folder = os.path.join(train_dir, label)
    img_files = [f for f in os.listdir(folder) if f.endswith(".png")]

    print(f"🔁 增强类别 {label}，共 {len(img_files)} 张图像")
    for f in tqdm(img_files):
        path = os.path.join(folder, f)
        if label == '0':  # 无病类别，增强两倍
            save_augmented(path, folder, label, num_augments=2)
        else:             # 有病类别，增强一次
            save_augmented(path, folder, label, num_augments=1)

print("✅ 所有增强图像已保存完毕！")
