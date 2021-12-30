

#　介绍
- 关于MTCNN（Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks）
* 地址：https://github.com/kpzhang93/MTCNN_face_detection_alignment
* 这个方法很经典
* protobuf，是google开发的序列化框架，通过该框架可以对数据进行序列化并进行传输
* caffemodel是caffe用来存储模型的文件格式，二进制权重存储格式
* prototxt是caffe的网络图结构存储的文件格式，文本格式存储，通过按照图结构读取后，使用caffemodel填充对应权重，进而可以推理
* 读论文和其他科普文章



- 作业：导入权重，并运行网络

## 读取图片
img = cv2.imread('my.jpg')
img = cv2.resize(img,dsize=(12,12))
plt.imshow(img[:,:,::-1])
img.shape

# 上课思路
写class PNet，因为作业已经说明了这是一个网络。结合prototxt我们可以知道它的结构
写到load

# 全卷积的介绍
    - 理解成是滑窗

#　数据预处理
    - 数据预处理 
        - imread后就要：!!!!!!
        - input_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        - input_image = (input_image - 127.5) * 0.0078125   
        - transpose(2,1,0)[None]   hwc --> cwh   ref: https://www.mathworks.com/help/coder/ug/what-are-column-major-and-row-major-representation-1.html


# FCN.ipynb
# NMS
# Pyramid
    到此为止，我们知道了PNet是怎么工作的。它是如何与RNet相连接的呢

# crop_resize_to_with_warpaffine

# MTCNN_complete.py
    - 先过一遍 RNet 和 ONet
    - 完善PNet
    - 增加 class Net   class CaffeModelLoader  class RNet    class ONet   class MTCNN 等   
    - 


