# CS-577-DeepLearning-FinalProject

## Project Description
A low-light environment is an essential element of our daily lives. As the day turns to night, the amount of available light diminishes, making our surroundings darker and, as a result, reducing our ability to complete even minor tasks owing to the loss of visibility. Computer vision research and systems targeted at supporting individuals in their daily tasks, as well as increasing safety and security, could be extremely beneficial in such circumstances. However, most low-light research focuses on image augmentation, which has no relevance to assistive systems, or night vision surveillance, which necessitates expensive hardware, while more related fields such as object recognition receive little attention. 

Because of the intrinsic complexity of real-life situations, object detection is a difficult challenge. It involves accurately recognizing and localizing objects within an image or video frame. The variety in item appearance, scale, orientation, occlusion, and ambient parameters contributes to this complexity. Objects can take on a variety of shapes, colors, and sizes, making it challenging for detection algorithms to generalize well. Objects can also overlap, occlude one another, or merge into complex backdrops, adding to the difficulty. 

Detecting objects in low-light circumstances is difficult due to decreased visibility and the restricted amount of information available to computer vision systems. In such cases, the acquired images frequently lack contrast and clarity, making it difficult to differentiate things from their surroundings. This lack of illumination can lead to increased noise, loss of significant details, and lower object identification algorithm accuracy. Furthermore, shadows and reflections can complicate the task by producing false positives or hiding items of interest. To address these issues, low-light-specific hardware and algorithms are necessary, which frequently involve the use of advanced sensors and image enhancement techniques to improve the accuracy and reliability of object detection under these conditions. 

This project aims to compare and evaluate the performance of transformer-based and traditional deep-learning object detection models on different image enhancement techniques.

## Dataset - Exclusively-Dark-Image-Dataset
The Exclusively Dark [(ExDark)](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) dataset contains the largest collection of natural low-light images taken in visible light to date, including object level annotation. 
### Dataset Folders Structure
In git's repository root folder:
* [./ExDark/ExDark](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view?usp=sharing) - Original images from ExDark's git repository, subfolers for categories
* [./ExDark_Annno/ExDark_Annno](https://drive.google.com/file/d/1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i/view?usp=sharing) - Original annotations from ExDark's git repository, subfolders for categories
* [./ExDark_All](https://drive.google.com/drive/folders/12HhqaCy_45DxXQ6t49S3_It4mKBIsFHy?usp=sharing) - All images and annotations without subfolders
* [./ExDark_COCO](https://drive.google.com/drive/folders/1sabKOSHui8G8cI4w_lLxb4outId2YV_r?usp=sharing) - .JSON files for COCO format dataset generator

## Documents
* Project Intermediate Report - [Word Document](https://iit0-my.sharepoint.com/personal/pvidhyaravikumar_hawk_iit_edu/Documents/DL_Intermediate%20Project%20Report.docx?d=w016b9bc6dead47829f1876795bf3bb6e&csf=1&web=1&e=LKwkTV)
* Project Presentation - [Google PPT Document](https://docs.google.com/presentation/d/1wyljypQYRHxmpP_kKwDI-fUVvFBJUGjmEKa7qZp6xbk/edit?usp=sharing)
* Project Final Report - [Word Document](https://iit0-my.sharepoint.com/personal/pvidhyaravikumar_hawk_iit_edu/Documents/OBJECT%20DETECTION%20UNDER%20LOW-LIGHTING%20CONDITIONS.docx?d=w16f0c23dae97471390d5f54b601c5b1f&csf=1&web=1&e=mkvVbQ)

## Authors
* Ignacio Gomez Valverde (A20552714)
* Prashanth V.R. (A20531508)

## References
### Single Shot Detector
* [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
* [PyTorch SSD](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/)
### Transformer
* [End-to-End Detection with Transformers](https://arxiv.org/abs/2005.12872)
* [Detection-Transformer](https://github.com/AarohiSingla/Detection-Transformer/tree/main)
* [Using Custom Datasets to train DETR for object detection](https://medium.com/@soumyajitdatta123/using-custom-datasets-to-train-detr-for-object-detection-75a6426b3f4e)
