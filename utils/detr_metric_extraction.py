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
    test_path = "../ExDark_COCO/3200-1800-rest/test_set.json"
    model_path = "../Models/Transformer/lightning_logs/clahe/"

    # Initialization
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    test_set = CocoDetection(image_directory_path=imgs_path,  annotation_file_path=test_path, image_processor=image_processor)
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    results = {'images': [],
                'original_boxes': [],
                'original_labels': [],
                'predicted_labels': [],
                'predicted_boxes': [],
                'predicted_scores': []}

    # Loop
    for i in tqdm(range(len(test_set))):

        # Original Image
        _, anno = test_set[i]
        size = anno['size']
        cv_image = cv2.imread(os.path.join(imgs_path, test_set.coco.loadImgs(i)[0]['file_name']))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Prediction
        results_ = None
        with torch.no_grad():
            inputs = image_processor(images=cv_image, return_tensors='pt').to(device)
            outputs = model(**inputs)
            results_ = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=0.001, 
                target_sizes=[size]
            )[0]

        # Save
        results['images'].append(test_set.coco.loadImgs(i)[0]['file_name'])
        results['original_boxes'].append(anno['boxes'].tolist())
        results['original_labels'].append(anno['class_labels'].tolist())
        for j in range(len(results_["scores"])):
            if results_["scores"][j] >= 0.85:
                results['predicted_boxes'].append(results_['boxes'][j].tolist())
                results['predicted_labels'].append(results_['labels'][j].tolist())
                results['predicted_scores'].append(results_['scores'][j].tolist())

        # Save as json
        with open(f'../Models/Transformer/lightning_logs/{name}/results.json', 'w') as fp:
            json.dump(results, fp)

if __name__ == "__main__":
    name = input("Enter name: (clahe,...)")
    pred_json(name)