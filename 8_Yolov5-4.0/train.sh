#!/bin/bash
python train.py \
    --data data/voc.yaml \
    --cfg models/yolov5s_for_voc2007.yaml \
    --weights weights/yolov5s.pt \
    --batch-size 32 \
    --epochs 200