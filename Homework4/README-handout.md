# 几何计算前沿 第四次作业

本次作业满分15分，截止日期为2025.6.4，需要使用神经网络完成从点云重建表面的任务

### 任务

实现一个基于MLP的从点云重建表面的模型，训练数据是表面上的点云、空间中的采样点和对应SDF值，输出重建后的形状。

MLP的功能是输出空间中任意查询点的SDF值。假设输入的点云为$P$，查询点为$q \in Q$，则有：
$$
\tilde{F}(q) = \text{MLP}(q)
$$
训练时的损失函数可以设置为：
$$
L_{sdf} = \sum_{q_i \in Q} \left(
    \| \tilde{F}(q_i) - F(q_i) \|^2 + \lambda \| \nabla \tilde{F}(q_i) - N(q_i) \|^2
\right)
$$
其中$\tilde{F}(q_i)$为MLP预测的SDF值，$F(q_i)$为ground truth的SDF值，$\nabla \tilde{F}(q_i)$为SDF相对$q_i$的梯度（可以通过`torch.autograd.grad`计算），$N(q_i)$为ground truth的梯度，$\lambda$为超参数，控制SDF值和梯度的损失权重，可以自行设置。

测试时在空间中划分均匀格点，计算每个格点的SDF值，然后使用Marching Cubes算法提取mesh。

### 数据

数据在data文件夹下，为修复后的ShapeNet V1中的airplane类的5个样本。

每个uid文件夹下包括：

```
pointcloud.npz
  - "points": 表面上的点云坐标
  - "normals": 表面上的法向量
sdf.npz
  - "points": 空间中的采样点坐标
  - "grad": 采样点对应的SDF梯度
  - "sdf": 采样点对应SDF值
uid.obj: 用于参考的ground truth三维形状
```

### 拓展任务

MLP会更倾向于生成光滑的结果，使得重建表面的细节不足。

实现一个基于Fourier feature的position encoding，将$q$的坐标进行Fourier特征映射后再输入MLP，比较和原方法的结果。

> 参考论文：Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains

### 提交说明

- 代码：需要包含完整的训练和测试代码，并附一个README文档说明代码的运行方式
- 报告：需要包含以下内容
  - 神经网络模型和方法细节，如网络结构、参数量等
  - 训练策略，如学习率、优化器、训练轮数、超参数的设置等
  - 训练和重建所用的时间
  - 对五个模型的重建结果，与ground truth的对比
  - 如果做了拓展任务，还需要对比分析两种方法的效果
- 运行结果：Marching Cubes算法提取出的mesh，存储为obj或ply格式
- 以上内容打包在教学网提交
- 无需交训练数据和模型参数文件

### 评分标准

- 基础任务共11分
  - 代码实现6分，要求代码可以正确运行，重建出表面形状
  - 报告内容3分，模型和方法细节1分、训练策略和时间1分、重建结果分析1分
  - 重建效果2分，模型应当能够重建出与ground truth相似的表面形状
- 拓展任务共4分
  - 代码实现2分，要求代码可以正确运行，重建出表面形状
  - 报告内容1分，对两种方法的结果进行对比分析
  - 重建效果1分，模型应当能够重建出与ground truth相似的表面形状，并且在高频细节上与基础方法相比有所提升


