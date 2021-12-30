./
|-- Dockerfile
|-- LICENSE
|-- README.md
|-- __pycache__
|   `-- test.cpython-38.pyc
|-- bug_log.md
|-- comments_version
|-- data 
|   |-- coco.yaml    # coco 数据集配置文件
|   |-- coco128.yaml # coco128数据集配置文件
|   |-- hyp.finetune.yaml # 超参数微调配置文件
|   |-- hyp.scratch.yaml  # 超参数起始配置文件
|   |-- images            # images for inference     
|   |-- scripts           # 获取voc 和 coco 数据集 的下载命令
|   `-- voc.yaml          # voc数据集的配置文件
|-- ddp.sh                     
|-- ddp_tutorial.md
|-- detect.py
|-- hubconf.py
|-- imgs 
|-- kill.sh
|-- lab_report.md
|-- models
|   |-- __init__.py
|   |-- __pycache__     
|   |-- common.py       # 常见模型组件定义代码
|   |-- experimental.py # 实验性质的代码
|   |-- export.py       # 模型导出文件
|   |-- hub             # 
|   |-- yolo.py
|   |-- yolov5l.yaml    # yolov5l模型配置文件
|   |-- yolov5m.yaml
|   |-- yolov5s.yaml
|   |-- yolov5s_for_voc2007.yaml
|   `-- yolov5x.yaml
|-- prj_structure.md
|-- requirements.txt   # 环境要求
|-- runs               # 训练结果
|   `-- train
|-- target_class_vis.jpg
|-- test.py
|-- train.py
|-- train.sh
|-- tutorial.ipynb
|-- utils
|   |-- __init__.py
|   |-- __pycache__
|   |-- activations.py
|   |-- autoanchor.py
|   |-- datasets.py
|   |-- general.py        # 项目通用函数代码
|   |-- google_app_engine 
|   |-- google_utils.py   # google 云使用相关代码
|   |-- loss.py
|   |-- metrics.py
|   |-- plots.py
|   `-- torch_utils.py    # 辅助程序代码
|-- weights
|   |-- download_weights.sh
|   |-- yolov5s.onnx
|   `-- yolov5s.pt
`-- yolov5-3.1_comment_version.zip