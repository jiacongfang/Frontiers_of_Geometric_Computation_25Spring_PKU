# 几何计算前沿 第三次作业

本次作业满分15分，截止日期为2025.5.13，需要使用神经网络完成三维模型的分类任务

### 任务

请根据自身的计算资源，任意选择两个3D Classification的论文，复现其在[ModelNet40](https://modelnet.cs.princeton.edu/)数据集上的分类结果。

本次作业中可以使用论文开源代码仓库中的代码。

### 评分标准

满分15分，其中
- 两篇论文的复现各6分
  - 代码实现3分，需要训练和测试代码能够正确运行
  - 方法介绍1分，在报告中简要介绍论文的核心思想和方法
  - 结果复现2分，在报告中展示训练过程中的Loss和Accuracy曲线，并将最终测试结果与原论文中所汇报的结果进行对比
- 在报告中对比分析不同方法的参数量、运行速度、分类效果等，3分

### 提交要求

- 代码和报告打包提交，并附一个README文件说明代码的运行方式，包括数据预处理（如有）、训练和测试
- 需要提交训练好的模型参数，如果参数文件过大，教学网不方便上传，可以上传北大网盘，并在报告中附下载链接
- **请不要提交数据集**

### 参考资料

- OCNN: https://github.com/octree-nn/ocnn-pytorch
- PointNet: https://github.com/charlesq34/pointnet
- PointNet++: https://github.com/charlesq34/pointnet2
- PointCNN: https://github.com/yangyanli/PointCNN
- PointConv: https://github.com/DylanWusee/pointconv
- Point Transformer: https://github.com/POSTECH-CVLab/point-transformer
- 其他感兴趣的方法

