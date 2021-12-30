# Multi-GPU DDP
## world size and rank
ref: [world size and rank](https://stackoverflow.com/questions/58271635/in-distributed-computing-what-are-world-size-and-rank#:~:text=Local%20rank%20is%20the%20a,5%2C%206%2C%207%5D%20)
- world: all processes. Ususally one gpu one process
- world size: the number of processes for your training
- Rank: the unique id given to a process
- e.g.
    - 2 servers or nodes
    - each with 4 gpus
    - the world size = 2 x 4 = 8
    - ranks = [0, 1, 2, 3, 4, 5, 6, 7]
    - in each node, the local rank is [0, 1, 2, 3]

## python m flag
```bash
CUDA_VISIBLE_DEVICE=1,2 python -m \
torch.distributed.launch --nproc_per_node = 2 --master_port = 55678 \
train.py --batch_size = 4 --about = MY_YOLOV5_TRAIN
```
It amounts to say we start with running the torch.distributed.launch.py then we run the train.py.

When running torch.distributed.launch.py, we pass --npproc_per_node, --master_port, 

for train.py, we pass --batch_size, --about.

## argument description
- nproc_per_node : num of processes on each server
- mater_port : a ddp process should have an unique master_port. Otherwise, the error of address already in use will be raised.
- OMP_NUM_THREADS=2 : OpenMP(c++) argument. If not passed, some annoying information will be displayed.
