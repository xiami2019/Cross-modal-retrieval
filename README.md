# Cross-modal-retrieve
By Xiami2019

## Dataset  
Images and text are from 'iaprtc-12'https://www.imageclef.org/photodata  
Labels are from 'saiaprtc-12'https://www.imageclef.org/SIAPRdata  
And make a mixture dataset for cross-modal retrieve by simply concatenate the labels, images, text.  
In this dataset, we have 20000 images and their corresponding description both in English and German.  
There are also multi labels for each image-text pair.
Follow the setting in reference paper, I choose 10000 images as train set.  
When testing, I choose 2000 images as query set and the last 18000 images as database.

## Model
The network consists of image model and text model. I use CNN networks pretrained Resnet18 as the image model and pretrained BERT-base as the text model.

## Objective
Final objective consists of four triplet losses：  
F=L_{I\rightarrow T}+L_{T\rightarrow I}+L_{T\rightarrow T}+L_{I\rightarrow I}

## Result:  
Due to time and devices limited, extensive experient has not been executed.
After 200 epoch  
Text→Images  

language | 16bits | 32bits | 48bits | 64bits  
|:---: |:---: |:---: | :---: |:---: |  
`English` | **To be added** | **0.4583** | **To be added** | **To be added**  
`German` | **To be added** | **0.4514** | **To be added** | **To be added**  

Images→Text  

language | 16bits | 32bits | 48bits | 64bits  
|:---: |:---: |:---: | :---: |:---: |  
`English` | **To be added** | **0.4533** | **To be added** | **To be added**  
`German` | **To be added** | **0.4421** | **To be added** | **To be added**  
