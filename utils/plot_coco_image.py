import os
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw   

def plot_coco_image(coco_dataset, imgs_path):

    # Category ID to Label Name
    id2label = {k: v["name"] for k, v in coco_dataset.coco.cats.items()}

     # Colors (Define colors for each class)
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'lime', 'teal', 'lavender']

    # Image
    image_ids = coco_dataset.coco.getImgIds()
    image_id = random.choice(image_ids)
    image = coco_dataset.coco.loadImgs(image_id)[0]
    print("Image ID: ", image_id, "Image: ", image["file_name"])

    # Image Annotations
    annotation_ids = coco_dataset.coco.getAnnIds(imgIds=image_id)
    annotations = coco_dataset.coco.loadAnns(annotation_ids)
    print("\nAnnotations: ")
    for anno in annotations:
        print(anno, 'Category Label: ', id2label[anno['category_id']])

    # Plot Image
    img_path = os.path.join(imgs_path, image["file_name"])
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for anno in annotations:
        cat_id = anno['category_id']
        bbox = [int(b) for b in anno["bbox"]]
        draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], width=3, outline=colors[cat_id])
        draw.text([bbox[0], bbox[1]], id2label[cat_id], fill=colors[cat_id])
    plt.imshow(img)
    plt.show()