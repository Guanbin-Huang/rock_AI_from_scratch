- 为什么我们要引入CNN（传统的神经网络（MLP/BP）跟CNN相比的不同）
    - ref : https://blog.csdn.net/weixin_44177568/article/details/102812050 (一定读)
    - ref: 关于平移不变性： https://zhuanlan.zhihu.com/p/38024868 (想说明的是深度学习里很多问题都是没有特别定论的说法的，存在争议)


- 卷积介绍与实现
    - 概念： CNN精简版.pptx 
    - 实现： 0.conv_op.py

- CNN 的直观理解
    - cnn_intuition.jpg  特征提取器+分类器


- 工业级实现：
    - GEMM
        ref: (瞄一眼意思意思即可)
            https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/
            https://blog.csdn.net/dwyane12138/article/details/78449898

        
        - conv -> GEMM: 原因
            - 1. 乘法分散，效率很低
            - 2. 乘法尽量密集一些，矩阵已经被很多计算机科学家去优化了
        - im2col and col2im 的实现
    
    - Winograd
    - FFT 快速傅里叶变换实现卷积

- 作业
    - 实现输入为2张3通道图，输出为2通道结果。kernel采用3x3第一组采用拉普拉斯核[1, 1, 1, 1, -8, 1, 1, 1, 1]，第二组采用梯度核[1, 0, -1, 1, 0, -1, 1, 0, -1]
    - 实现两张图，两组卷积，两张图都resize成5x5


- 参考资料
    slide/ppt
    onenote tutorial 卷积
    ipad 2.conv
    self-made-conv文件夹
    main.py
    等



