# Add Utils to path
import os
import cv2
import json
import torch
from tqdm import tqdm 
from datasets_generators import CocoDetection

# PyTorch
from transformers import DetrImageProcessor, DetrForObjectDetection

def pred_json(name):

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    imgs_path = "../ExDark_All/Images"
    test_path = "../ExDark_COCO/test_set.json"
    model_path = "../Models/Transformer/lightning_logs/clahe/"

    # Initialization
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    test_set = CocoDetection(image_directory_path=imgs_path,  annotation_file_path=test_path, image_processor=image_processor)
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    results = {"annotations": []}
    
    # Loop
    image_id = 0
    for i in tqdm(range(len(test_set))):

        # Original Image
        _, anno = test_set[i]
        file_name = test_set.coco.loadImgs(i)[0]['file_name']
        original_size = anno['orig_size']
        cv_image = cv2.imread(os.path.join(imgs_path, file_name))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Prediction
        results_ = None
        with torch.no_grad():
            inputs = image_processor(images=cv_image, return_tensors='pt').to(device)
            outputs = model(**inputs)
            results_ = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=0.001, 
                target_sizes=[original_size]
            )[0]

        for j in range(len(results_["scores"])):
            if results_["scores"][j] >= 0.85:
                bbox = results_["boxes"][j].tolist()
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                bbox = [float(b) for b in bbox]
                results['annotations'].append({"image_id": image_id,
                                               "category_id": results_["labels"][j].item(),
                                               "bbox": bbox,
                                               "score": results_["scores"][j].item(),})	
        image_id += 1

    # Save as json
    with open(f'../Models/Transformer/lightning_logs/{name}/results.json', 'w') as fp:
        json.dump(results, fp)

if __name__ == "__main__":
    name = input("Enter name: (clahe,...)\n>")
    pred_json(name)