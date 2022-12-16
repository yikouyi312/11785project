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
* [Test](#Test)


## Data Preparation 
See Section 3.2.1 Data Preparation and 3.2.2 Classification, 
1. sample two $w^+$ latent code set $D_0$ and $D_{noise}$.
     - take dataset in ./training_runs/dataset, select 5%-10% data from HHFQ 1024 x 1024 (self.num)
     - create files to ./training_runs/dataset/{args.dataset_name}
          - D0: without noise $D_0$
          - Dnoise: add noise $D_{noise}$
2. get hair scores through a hair classifier and gender scores through a gender classifier for both $D_0$ and $D_{noise}$.
     - wp.npy, w.npy, z.npy, hair_scores.npy and gender_scores.npy
          - hair_scores, 1: not bald, 0: bald 
          - gender_score, 1: male, 0: female 
       
## Boundary Training
See 3.2.3 Separation Boundary Training

Apply the separation boundary training algorithm in InterFaceGAN to train two separation boundaries in the latent space of StyleGAN2-ada to provide general directions for male hair removal and gender transformation.

- Option one: Train male hair separation boundary by InterfaceGAN on $D_0$
     - need $D_0$ have enough bald male results.

- Option two: Use the pretrained hair separation boundary

## Male hair remove
See 3.2.4 Male Hair Removal and Training, equation (1) and (2), and optimize the latent code by minimizing the full
loss function $\mathcal{L}_{dif}$ obtain $\hat{w}^{*+}_m$

Training bald male data using hair boundary, latent_space_type = 'wp'

     create files in './training_runs/male_training'
     - mask: save hair_mask (including hat) 
     - res_wp_codes, 
     - res_img: viz_result(image after synthesis and diffusion)

## Initial Model Training
See 3.2.4 Male Hair Removal and Training, composing pairs of male latent codes $w^+_m$ and $\hat{w}^+_m$ to construct dataset $H_m$ and training a fully connected network $M_m$.

1. construct dataset $H_m$
     - Create files to ./training_runs/{args.mapper_name}/data
          - train.txt
          - val.txt
          - test.txt
3. traine network $M_m$.



Train model for man


## Female hair remove

## Final Model Training

## Test
