#!/bin/bash
export OMP_NUM_THREADS=1  # speeds up MinkowskiEngine
export CUDA_VISIBLE_DEVICES=5
export DETECTRON2_DATASETS="/projects/katefgroup/language_grounding/SEMSEG_100k"

CURR_DBSCAN=0.95
CURR_TOPK=500
CURR_QUERY=150

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="2d" \
data/datasets=scannet_2d \
general.eval_on_segments=true \
general.train_on_segments=true

# TEST
# python main_instance_segmentation.py \
# general.experiment_name="scannet_2d_validation_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}" \
# general.project_name="scannet_2d_eval" \
# general.checkpoint='/home/pkatara/Mask3D/saved/2d/last-epoch.ckpt' \
# general.train_mode=false \
# general.eval_on_segments=true \
# general.train_on_segments=true \
# model.num_queries=${CURR_QUERY} \
# general.topk_per_image=${CURR_TOPK} \
# general.use_dbscan=true \
# general.dbscan_eps=${CURR_DBSCAN} \
# data/datasets=scannet_2d