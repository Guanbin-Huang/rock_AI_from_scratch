- ddp not working
    - switch between ddp.sh and train.sh
        - required to clear the cache in /datav/shared/dataset/VOC/labels and generate the cache again


- ./ddp.sh: line 6: --name: command not found
    - no space after back slash \


- yolo lab
AttributeError: Can't get attribute 'SPPF' on <module 'models.common' from '/datav/shared/hgb/shenlan-yolov5-5.0/models/common.py'>
    - yolo version problem



- (base) root@0fe6dad34dfa:/datav/shared/hgb/yolov5-4.0-for-learning# ./ddp.sh  (clear the cache)
    ```bash
    /root/anaconda3/lib/python3.8/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
    and will be removed in future. Use torchrun.
    Note that --use_env is set by default in torchrun.
    If your script expects `--local_rank` argument to be set, please
    change it to read from `os.environ['LOCAL_RANK']` instead. See 
    https://pytorch.org/docs/stable/distributed.html#launch-utility for 
    further instructions

    warnings.warn(
    WARNING:torch.distributed.run:
    *****************************************
    Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
    *****************************************
    Using torch 1.10.0+cu113 CUDA:0 (NVIDIA GeForce RTX 3090, 24268.3125MB)
                            CUDA:1 (NVIDIA GeForce RTX 3090, 24268.3125MB)
                            CUDA:2 (NVIDIA GeForce RTX 3090, 24268.3125MB)
                            CUDA:3 (NVIDIA GeForce RTX 3090, 24268.3125MB)


    Added key: store_based_barrier_key:1 to store for rank: 0
    Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
    Namespace(adam=False, batch_size=8, bucket='', cache_images=False, cfg='models/yolov5s_for_voc2007.yaml', data='data/voc.yaml', device='', epochs=200, evolve=False, exist_ok=False, global_rank=0, hyp='data/hyp.scratch.yaml', image_weights=False, img_size=[640, 640], local_rank=0, log_artifacts=False, log_imgs=16, multi_scale=False, name='hgb_yolov5_by', noautoanchor=False, nosave=False, notest=False, project='runs/train', quad=False, rect=False, resume=False, save_dir='runs/train/hgb_yolov5_by2', single_cls=False, sync_bn=False, total_batch_size=32, weights='weights/yolov5s.pt', workers=8, world_size=4)
    Start Tensorboard with "tensorboard --logdir runs/train", view at http://localhost:6006/
    Hyperparameters {'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0, 'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0}

                    from  n    params  module                                  arguments                     
    0                -1  1      3520  models.common.Focus                     [3, 32, 3]                    
    1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
    2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
    3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
    4                -1  1    156928  models.common.C3                        [128, 128, 3]                 
    5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
    6                -1  1    625152  models.common.C3                        [256, 256, 3]                 
    7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
    8                -1  1    656896  models.common.SPP                       [512, 512, [5, 9, 13]]        
    9                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
    10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
    11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
    12           [-1, 6]  1         0  models.common.Concat                    [1]                           
    13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
    14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
    15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
    16           [-1, 4]  1         0  models.common.Concat                    [1]                           
    17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
    18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
    19          [-1, 14]  1         0  models.common.Concat                    [1]                           
    20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
    21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
    22          [-1, 10]  1         0  models.common.Concat                    [1]                           
    23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
    24      [17, 20, 23]  1     67425  models.yolo.Detect                      [20, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
    /root/anaconda3/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)
    return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    Model Summary: 283 layers, 7114785 parameters, 7114785 gradients, 16.5 GFLOPS

    Transferred 354/362 items from weights/yolov5s.pt
    Scaled weight_decay = 0.0005
    Optimizer groups: 62 .bias, 62 conv.weight, 59 other
    Scanning '/datav/shared/dataset/VOC/labels/train.cache' for images and labels... 16551 found, 0 missing, 0 empty, 0 corrupted: 100%|████████████████████████████| 16551/16551 [00:00<?, ?it/s]
    Traceback (most recent call last):
    File "train.py", line 519, in <module>
        train(hyp, opt, device, tb_writer, wandb)
    File "train.py", line 189, in train
        dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
    File "/datav/shared/hgb/yolov5-4.0-for-learning/utils/datasets.py", line 62, in create_dataloader
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
    File "/datav/shared/hgb/yolov5-4.0-for-learning/utils/datasets.py", line 385, in __init__
        labels, shapes = zip(*cache.values())
    TypeError: 'float' object is not iterable
    /root/anaconda3/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)
    return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    /root/anaconda3/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)
    return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    /root/anaconda3/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)
    return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 52837 closing signal SIGTERM
    WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 52842 closing signal SIGTERM
    WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 52843 closing signal SIGTERM
    ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 52833) of binary: /root/anaconda3/bin/python
    Traceback (most recent call last):
    File "/root/anaconda3/lib/python3.8/runpy.py", line 194, in _run_module_as_main
        return _run_code(code, main_globals, None,
    File "/root/anaconda3/lib/python3.8/runpy.py", line 87, in _run_code
        exec(code, run_globals)
    File "/root/anaconda3/lib/python3.8/site-packages/torch/distributed/launch.py", line 193, in <module>
        main()
    File "/root/anaconda3/lib/python3.8/site-packages/torch/distributed/launch.py", line 189, in main
        launch(args)
    File "/root/anaconda3/lib/python3.8/site-packages/torch/distributed/launch.py", line 174, in launch
        run(args)
    File "/root/anaconda3/lib/python3.8/site-packages/torch/distributed/run.py", line 710, in run
        elastic_launch(
    File "/root/anaconda3/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 131, in __call__
        return launch_agent(self._config, self._entrypoint, list(args))
    File "/root/anaconda3/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 259, in launch_agent
        raise ChildFailedError(
    torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
    ============================================================
    train.py FAILED
    ------------------------------------------------------------
    Failures:
    <NO_OTHER_FAILURES>
    ------------------------------------------------------------
    Root Cause (first observed failure):
    [0]:
    time      : 2021-10-30_09:16:25
    host      : 0fe6dad34dfa
    rank      : 0 (local_rank: 0)
    exitcode  : 1 (pid: 52833)
    error_file: <N/A>
    traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
    ```
