import os
import cv2
import shutil
from tqdm import tqdm
from image_enhancement_functions import clahe, color_balance_adjustment, histogram_equalization

# Main Folders
imgs = "../ExDark/ExDark"
annos = "../ExDark_Annno/ExDark_Annno"
all_imgs = "../ExDark_All/Images"
all_annos = "../ExDark_All/Annotations"

print("Creating folders...")

# Define folders to be created
raw = "../ExDark_YOLO/RAW"
raw_train = "../ExDark_YOLO/RAW/train"
raw_train_images = "../ExDark_YOLO/RAW/train/images"
raw_train_labels = "../ExDark_YOLO/RAW/train/labels"
raw_valid = "../ExDark_YOLO/RAW/valid"
raw_valid_images = "../ExDark_YOLO/RAW/valid/images"
raw_valid_labels = "../ExDark_YOLO/RAW/valid/labels"
clahe_p = "../ExDark_YOLO/CLAHE"
clahe_train = "../ExDark_YOLO/CLAHE/train"
clahe_train_images = "../ExDark_YOLO/CLAHE/train/images"
clahe_train_labels = "../ExDark_YOLO/CLAHE/train/labels"
clahe_valid = "../ExDark_YOLO/CLAHE/valid"
clahe_valid_images = "../ExDark_YOLO/CLAHE/valid/images"
clahe_valid_labels = "../ExDark_YOLO/CLAHE/valid/labels"
cb = "../ExDark_YOLO/CB"
cb_train = "../ExDark_YOLO/CB/train"
cb_train_images = "../ExDark_YOLO/CB/train/images"
cb_train_labels = "../ExDark_YOLO/CB/train/labels"
cb_valid = "../ExDark_YOLO/CB/valid"
cb_valid_images = "../ExDark_YOLO/CB/valid/images"
cb_valid_labels = "../ExDark_YOLO/CB/valid/labels"
he = "../ExDark_YOLO/HE"
he_train = "../ExDark_YOLO/HE/train"
he_train_images = "../ExDark_YOLO/HE/train/images"
he_train_labels = "../ExDark_YOLO/HE/train/labels"
he_valid = "../ExDark_YOLO/HE/valid"
he_valid_images = "../ExDark_YOLO/HE/valid/images"
he_valid_labels = "../ExDark_YOLO/HE/valid/labels"
test = "../ExDark_YOLO/test"
test_images = "../ExDark_YOLO/test/images"
test_labels = "../ExDark_YOLO/test/labels"

# Create folders
os.makedirs(raw, exist_ok=True)
os.makedirs(raw_train, exist_ok=True)
os.makedirs(raw_train_images, exist_ok=True)
os.makedirs(raw_train_labels, exist_ok=True)
os.makedirs(raw_valid, exist_ok=True)
os.makedirs(raw_valid_images, exist_ok=True)
os.makedirs(raw_valid_labels, exist_ok=True)
os.makedirs(clahe_p, exist_ok=True)
os.makedirs(clahe_train, exist_ok=True)
os.makedirs(clahe_train_images, exist_ok=True)
os.makedirs(clahe_train_labels, exist_ok=True)
os.makedirs(clahe_valid, exist_ok=True)
os.makedirs(clahe_valid_images, exist_ok=True)
os.makedirs(clahe_valid_labels, exist_ok=True)
os.makedirs(cb, exist_ok=True)
os.makedirs(cb_train, exist_ok=True)
os.makedirs(cb_train_images, exist_ok=True)
os.makedirs(cb_train_labels, exist_ok=True)
os.makedirs(cb_valid, exist_ok=True)
os.makedirs(cb_valid_images, exist_ok=True)
os.makedirs(cb_valid_labels, exist_ok=True)
os.makedirs(he, exist_ok=True)
os.makedirs(he_train, exist_ok=True)
os.makedirs(he_train_images, exist_ok=True)
os.makedirs(he_train_labels, exist_ok=True)
os.makedirs(he_valid, exist_ok=True)
os.makedirs(he_valid_images, exist_ok=True)
os.makedirs(he_valid_labels, exist_ok=True)
os.makedirs(test, exist_ok=True)
os.makedirs(test_images, exist_ok=True)
os.makedirs(test_labels, exist_ok=True)

print("Splitting data...")

# Split data
train_data = []
valid_data = []
test_data = []

for label_dir in os.listdir(imgs):
    train_data.extend(os.listdir(os.path.join(imgs, label_dir))[:250])
    valid_data.extend(os.listdir(os.path.join(imgs, label_dir))[250:400])
    test_data.extend(os.listdir(os.path.join(imgs, label_dir))[400:])

print(f'  Split: {len(train_data)} - {len(valid_data)} - {len(test_data)}')

print("Defining labels...")

# Labels
labels = dict([(label, i) for i, label in enumerate(os.listdir(imgs))])
print(f'  Labels: {labels}')

print("Applying enhancements and copying images...")

# Copy images to folders
for img in tqdm(os.listdir(all_imgs)):
    
    # Read RAW image
    image = cv2.imread(os.path.join(all_imgs, img))

    # Train
    if img in train_data:
        # RAW
        shutil.copy(os.path.join(all_imgs, img), os.path.join(raw_train_images, img))

        # CLAHE
        clahe_image = clahe(image)
        cv2.imwrite(os.path.join(clahe_train_images, img), clahe_image)

        # CB
        cb_image = color_balance_adjustment(image)
        cv2.imwrite(os.path.join(cb_train_images, img), cb_image)

        # HE
        he_image = histogram_equalization(image)
        cv2.imwrite(os.path.join(he_train_images, img), he_image)

    # Valid
    elif img in valid_data:
        # RAW
        shutil.copy(os.path.join(all_imgs, img), os.path.join(raw_valid_images, img))

        # CLAHE
        clahe_image = clahe(image)
        cv2.imwrite(os.path.join(clahe_valid_images, img), clahe_image)

        # CB
        cb_image = color_balance_adjustment(image)
        cv2.imwrite(os.path.join(cb_valid_images, img), cb_image)

        # HE
        he_image = histogram_equalization(image)
        cv2.imwrite(os.path.join(he_valid_images, img), he_image)

    # Test
    else:
        shutil.copy(os.path.join(all_imgs, img), os.path.join(test_images, img))

print("Copying labels...")

# Copy labels to folders
for anno in tqdm(os.listdir(all_annos)):
    # Train
    if anno in train_data:
        shutil.copy(os.path.join(all_annos, anno), os.path.join(raw_train_labels, anno))
        shutil.copy(os.path.join(all_annos, anno), os.path.join(clahe_train_labels, anno))
        shutil.copy(os.path.join(all_annos, anno), os.path.join(cb_train_labels, anno))
        shutil.copy(os.path.join(all_annos, anno), os.path.join(he_train_labels, anno))

    # Valid
    elif anno in valid_data:
        shutil.copy(os.path.join(all_annos, anno), os.path.join(raw_valid_labels, anno))
        shutil.copy(os.path.join(all_annos, anno), os.path.join(clahe_valid_labels, anno))
        shutil.copy(os.path.join(all_annos, anno), os.path.join(cb_valid_labels, anno))
        shutil.copy(os.path.join(all_annos, anno), os.path.join(he_valid_labels, anno))

    # Test
    else:
        shutil.copy(os.path.join(all_annos, anno), os.path.join(test_labels, anno))

print("Creating YAML files...")

# Create YAML files
with open("../ExDark_YOLO/raw_train.yaml", "w") as f:
    f.write(f"train: {raw_train}\n")
    f.write(f"val: {raw_valid}\n")
    f.write(f"nc: {len(labels)}\n")
    f.write(f"names: {list(labels.keys())}")

with open("../ExDark_YOLO/clahe_train.yaml", "w") as f:
    f.write(f"train: {clahe_train}\n")
    f.write(f"val: {clahe_valid}\n")
    f.write(f"nc: {len(labels)}\n")
    f.write(f"names: {list(labels.keys())}")

with open("../ExDark_YOLO/cb_train.yaml", "w") as f:
    f.write(f"train: {cb_train}\n")
    f.write(f"val: {cb_valid}\n")
    f.write(f"nc: {len(labels)}\n")
    f.write(f"names: {list(labels.keys())}")

with open("../ExDark_YOLO/he_train.yaml", "w") as f:
    f.write(f"train: {he_train}\n")
    f.write(f"val: {he_valid}\n")
    f.write(f"nc: {len(labels)}\n")
    f.write(f"names: {list(labels.keys())}")

print("Done!")