## Step 1: Activate the environment
conda activate fdgl2
cd /mnt/data1/zhaoxinjian/zhongkai/TMLR_VISG_

## Step 2: Generate the dataset
python3 generate_dataset.py --dataset=PROTEINS --layout=spring
#--dataset: IMDB_MULTI, NCI1, PROTEINS
#--layout: spring, circular, kamada_kawai, random, shell, spectral

###############################################################################################################
############### seed 1 ### seed 2 ### seed 3 ### seed 4 ### seed 5 ### seed 6 ### seed 7 ### seed 8 ### seed 9 
#######gcn##### 6518   ### 7589   ### 6696   ### 6875   ### 6964   ### 7768   ### 7232   ### 6339   ### 7321
#vis-ConvSmall# 6964   ### 7411   ### 6964   ### 7411   ### 7679   ### 7411   ### 8036   ### 7500   ### 7857
#vis-ConvBase## 7232   ### 7232   ### 7232   ### 7143   ### 7143   ### 7321   ### 
#gcn-ConvSmall#
###############################################################################################################

## Step 3: Eval all models
CUDA_VISIBLE_DEVICES=1 python3 visgnn.py --dataset=PROTEINS --seed 3
#--dataset: IMDB_MULTI, NCI1, PROTEINS
#--model_type: resnet, vit, convnext_base
#--seed
#--batch_size, image_size, lr, epochs 
#--center_crop, horizontal_flip, rotation, affine, perspective, normalize

CUDA_VISIBLE_DEVICES=1 python3 /mnt/data1/zhaoxinjian/zhongkai/TMLR_VISG_/5_multi_modality/constr.py --seed 1
