## 11-785 Final Project

Hair-Removal Network based on StyleGAN.

Detail implementation in Train.ipynb


### Table of contents
* [Data Preparation](#Data_Preparation)
* [Boundary Training](#Boundary_Training)
* [Male hair remove](#Male_hair_remove)
* [Initial Model Training](#Initial_Model_Training)
* [Female hair remove](#Female_hair_remove)
* [Final Model Training](#Final_Model_Training)


## Data Preparation 
dataset in ./training_runs/dataset, select 5%-10% data from HHFQ 1024 x 1024

create files to ./training_runs/dataset/{args.dataset_name}
- wp.npy, w.npy, z.npy, hair_scores.npy and gender_scores.npy
     - hair_scores, 1: not bald, 0: bald 
     - gender_score, 1: male, 0: female 
     - D0: without noise
     - Dnoise: add noise


## Boundary Training
Apply the separation boundary training algorithm in InterFaceGAN to train two separation boundaries in the latent space of StyleGAN2-ada to provide general directions for male hair removal and gender transformation.


## Male hair remove



## Initial Model Training

## Female hair remove

## Final Model Training
