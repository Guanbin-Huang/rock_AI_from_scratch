#!/bin/bash
python -m torch.distributed.launch \
            --nproc_per_node 4 \
            --master_port 56678 \
        train.py \
            --name hgb_yolov5_by \
            --data data/voc.yaml \
            --cfg models/yolov5s_for_voc2007.yaml \
            --weights weights/yolov5s.pt \
            --batch-size 64 \
            --epochs 200
