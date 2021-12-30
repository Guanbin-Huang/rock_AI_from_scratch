- yolov5-4.0
- pytorch and cuda: 
    - Using torch 1.10.0+cu113 

- GPU:
    - CUDA:0 (NVIDIA GeForce RTX 3090, 24268.3125MB) x  4

- hyperameters:
    - 'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0, 'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0

- ddp.sh: \
--name hgb_yolov5_by \
--data data/voc.yaml \
--cfg models/yolov5s_for_voc2007.yaml \
--weights weights/yolov5s.pt \
--batch-size 64 \
--epochs 200


result:
- 5.75it/s  
- 6.17it/s  
- 6.19it/s