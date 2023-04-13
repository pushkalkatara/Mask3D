#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export CUDA_VISIBLE_DEVICES=5

CURR_DBSCAN=0.95
CURR_TOPK=500
CURR_QUERY=150

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="validation_overfit_2" \
general.eval_on_segments=true \
general.train_on_segments=true \
data.train_mode="validation"

# TEST
# python main_instance_segmentation.py \
# general.experiment_name="validation_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}" \
# general.project_name="scannet_eval" \
# general.checkpoint='/home/pkatara/Mask3D/saved/validation/last-epoch.ckpt' \
# general.train_mode=false \
# general.eval_on_segments=true \
# general.train_on_segments=true \
# model.num_queries=${CURR_QUERY} \
# general.topk_per_image=${CURR_TOPK} \
# general.use_dbscan=true \
# general.dbscan_eps=${CURR_DBSCAN}