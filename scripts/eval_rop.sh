export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
export PYTHONPATH=${PYTHONPATH}:incubator-mxnet/python/

TRAIN_DIR=/home/qileimail123/data0/RetinaImg/ROP_COCO/
DATASET=Retina
SET=train2014
TEST_SET=val2014

# Test
python eval_maskrcnn.py \
    --network resnet_fpn \
    --has_rpn \
    --dataset ${DATASET} \
    --image_set ${TEST_SET} \
    --prefix ${TRAIN_DIR}final \
    --result_path /home/qileimail123/data0/RetinaImg/ROP_COCO/mask_results/baseline_pred/ \
    --epoch 0 \
    --gpu 0

python data/cityscape/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py
