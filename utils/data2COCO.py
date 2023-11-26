import os
import json
import random

# Seed
random.seed(33)

# Initialize COCO json format
train_json = {
    "images": [],
    "annotations": [],
    "categories": [],
    }

val_json = {
    "images": [],
    "annotations": [],
    "categories": [],
    }

test_json = {
    "images": [],
    "annotations": [],
    "categories": [],
    }

# Paths
cat_dir = '../ExDark/ExDark'
anno_dir = '../ExDark_All/Annotations'
img_dir = '../ExDark_All/Images'
coc_dir = '../ExDark_COCO' # Change to your own path

# Categories (Subfolders)
categories = os.listdir(cat_dir)
for i, c in enumerate(categories):
    train_json["categories"].append({"id": i, 
                                    "name": c})
    val_json["categories"].append({"id": i,
                                    "name": c})
    test_json["categories"].append({"id": i,
                                    "name": c})
    
print('Categories Done')

# Path List
img_list = os.listdir(img_dir)
random.shuffle(img_list)

# Split
train_list = []
val_list = []
test_list = []
for subf in os.listdir(cat_dir):
    subdir = os.path.join(cat_dir, subf)
    imgs = os.listdir(subdir)
    train_list.extend(imgs[:250])
    val_list.extend(imgs[250:400])
    test_list.extend(imgs[400:])

# Print values
print('Length of train set: ', len(train_list))
print('Length of val set: ', len(val_list))
print('Length of test set: ', len(test_list))

print('Split Done')

# Train Json
img_id = 0
anno_id = 0
for img in train_list:
    # Image
    train_json["images"].append({"id": img_id,
                                 "file_name": img})
    # Annotations
    with open(os.path.join(anno_dir, img + '.txt'), 'r') as f:
        lines = f.readlines()[1:]
        for l in lines:
            train_json['annotations'].append({"id": anno_id,
                                              "image_id": img_id,
                                              "category_id": categories.index(l.split(' ')[0]),
                                              "bbox": l.split(' ')[1:5],
                                              "area": float(l.split(' ')[3])*float(l.split(' ')[4])})   
            # Update annotation id 
            anno_id += 1
    # Update image id
    img_id += 1

print('Train Set Done')

# Val Json
img_id = 0
anno_id = 0
for img in val_list:
    # Image
    val_json["images"].append({"id": img_id,
                               "file_name": img})
    # Annotations
    with open(os.path.join(anno_dir, img + '.txt'), 'r') as f:
        lines = f.readlines()[1:]
        for l in lines:
            val_json['annotations'].append({"id": anno_id,
                                            "image_id": img_id,
                                            "category_id": categories.index(l.split(' ')[0]),
                                            "bbox": l.split(' ')[1:5],
                                            "area": float(l.split(' ')[3])*float(l.split(' ')[4])})   
            # Update annotation id 
            anno_id += 1
    # Update image id
    img_id += 1

print('Val Set Done')

# Test Json
img_id = 0
anno_id = 0
for img in test_list:
    # Image
    test_json["images"].append({"id": img_id,
                                "file_name": img})
    # Annotations
    with open(os.path.join(anno_dir, img + '.txt'), 'r') as f:
        lines = f.readlines()[1:]
        for l in lines:
            test_json['annotations'].append({"id": anno_id,
                                             "image_id": img_id,
                                             "category_id": categories.index(l.split(' ')[0]),
                                             "bbox": l.split(' ')[1:5],
                                             "area": float(l.split(' ')[3])*float(l.split(' ')[4])})   
            # Update annotation id 
            anno_id += 1
    # Update image id
    img_id += 1

print('Test Set Done\nSaving...')

# Save Files
json.dump(train_json, open(os.path.join(coc_dir, 'train_set.json'), 'w'))
json.dump(val_json, open(os.path.join(coc_dir, 'val_set.json'), 'w'))
json.dump(test_json, open(os.path.join(coc_dir, 'test_set.json'), 'w'))

print('Done')