# Understanding Clouds from Satellite Images
### Disclaimer
This project was conducted for University of Toronto - School of Continuing Studies (SCS) as part of the Deep Learning 3546 Course. The dataset used for this project was retrieved from https://www.kaggle.com/c/understanding_cloud_organization

Submitted By:
 - Rueen Fiez
 - Nareshkumar Patel
 - Nisarg Patel

## Introduction
![Teaser Animation](assets/Teaser_AnimationwLabels.gif)


## Setup
Libraries used for the project:
- Mask R-CNN (V 2.1.0) - https://github.com/matterport/Mask_RCNN
- Keras (V 2.2.5) - https://github.com/keras-team/keras/
- TensorFlow (V 1.1.5) - https://github.com/tensorflow/tensorflow
- COCO Weights - http://cocodataset.org/#home

Folder Structure setup for training the model:</br>
```
SCS-DL-3546-Final-Project
│   assets (Git README images store directory)
│   mask_rcnn_cloudimages.h5 (Trained Weights so you don't need to train)
│   mask_rcnn_coco.h5 (COCO Weights)
│   Mask_RCNN (mask r-cnn code directory)
│   presentation
│   │   Cloud_Image_Classfication_Presentation.ppsx (Presentation show file)
│   │   Cloud_Image_Classfication_Presentation.pptx (Powerpoint file)
│   Cloud_Image_Classification.ipynb (Jupyter notebook / source code)
│   test_images
│   │   <All images for model testing>
│   │   # Note this is optional as the test set is not used.
│   train_images
│   │   <All images for model training & validation>
│   train.csv (annotation file that contains the masks for train images)
│   README.md (Readme file)
```

## Exploratory Data Analysis (EDA) 

![Empty_Non_Empty_Mask_Chart](assets/Empty_Non_Empty_Mask_Chart.png)

![Cloud_Types_Distribution](assets/Cloud_Types_Distribution.png)

![Num_labels_per_image](assets/Num_labels_per_image.png)

![Cloud_type_correlation](assets/Cloud_type_correlation.png)

## Mask R-CNN Model
**Model:** Mask R-CNN (Detection & Segmentation)</br>
**Weight:** Coco</br>
**Image Dimension:** 1400 x 2100 (H x W)</br>
**Steps Per Epoch:** 2,218</br>
**Validation Steps:** 555</br>
**Confidence:** 70% (minimum)</br>
**Heads Layer Epoch:**  1 (few as possible)</br>
**All Layer Epoch:** 5 (Hardware limitations)</br>
**Training Time:**  ~16 hrs (Colab - GPU)</br>
**Evaluation Metric:** Mean Average Precision (mAP)</br>

Here is a glimpse of train images right before the training process. This is what the Mask R-CNN model sees when its training its network.</br>
![Train_1_1](assets/Train_1_1.png)
![Train_1_2](assets/Train_1_2.png)

![Train_2_1](assets/Train_2_1.png)
![Train_2_2](assets/Train_2_2.png)

![Train_3_1](assets/Train_3_1.png)
![Train_3_2](assets/Train_3_2.png)

![Train_4_1](assets/Train_4_1.png)
![Train_4_2](assets/Train_4_2.png)

![Train_5_1](assets/Train_5_1.png)
![Train_5_2](assets/Train_5_2.png)

## Training & Validation Loss
![Train Loss](assets/loss.png)
*Training Loss*

![Val Loss](assets/val_loss.png)
*Validation Loss*

## Conclusion
![A1](assets/Actual_vs_pred_1.png)
![A2](assets/Actual_vs_pred_2.png)
![A3](assets/Actual_vs_pred_3.png)
![A4](assets/Actual_vs_pred_4.png)
![A5](assets/Actual_vs_pred_5.png)
![A6](assets/Actual_vs_pred_6.png)
![A7](assets/Actual_vs_pred_7.png)

## Model Evaluation

## Next Steps
1. Train the model for more epochs (100).
2. Use Image Augmentation with pre & post processing.
3. Experiment with different weights (Imagenet).
4. Experiment with different DL Packages (Py-torch) / Models (Faster R-CNN, U-Net).
5. Annotate images with segmentation masks.

### Presentation
- [PowerPoint](https://github.com/nishp763/SCS-DL-3546-Final-Project/blob/master/presentation/Cloud_Image_Classfication_Presentation.pptx)
- [PDF](https://github.com/nishp763/SCS-DL-3546-Final-Project/blob/master/Cloud_Image_Classfication_Presentation_COPY.pdf)
- [YouTube Video](https://youtu.be/wAOazvwSG5k)
