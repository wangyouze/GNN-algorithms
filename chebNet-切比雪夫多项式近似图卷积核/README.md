### Tensorflow-ChebNet-Tutorial

***

当GCN如日中天的时候，大部分人并不知道GCN其实是对ChebNet的进一步简化与近似，ChebNet与GCN都属于谱域上定义的图卷积网络。本教程将教你如何用Tensorflow构建ChebNet模型进行节点分类任务。完整的代码可在Github中下载：

### ChebNet简介

***

由图上的傅里叶变换公式我们可以得到卷积核h在图f上的卷积公式：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%28f*h%29_G%20%3D%20U%28%28U%5ETf%29%5Codot%20%28U%5ETh%29%29%20%3D%20Udiag%5B%5Chat%28h%29%28%5Clambda%20_1%29%2C...%2C%5Chat%28h%29%28%5Clambda%20_1%29%5DU%5ETf)

* 第一代GCN中简单把![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?diag%5B%5Chat%20h%28%5Clambda%20_1%29%2C...%2C%5Chat%20h%28%5Clambda%20_1%29%5D)中的对角线元素![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%20h%28%5Clambda%20_i%29)替换为参数![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctheta)，此时的GCN就变成了这个样子：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Csigma%20%28Udiag%5B%5Chat%7Bh%7D%28%5Clambda%20_1%29%2C...%2C%5Chat%7Bh%7D%28%5Clambda%20_N%29%20%5DU%5ETf%29%20%3D%20%5Csigma%20%28Ug_%5Ctheta%20U%5ETx%29)

  但是这样问题很多，如：

  1. 图卷积核参数量大，参数量与图中的节点的数量相同。

  2. 卷积核是全局的。

  3. 运算过程设计到特征分解，复杂度高。

* 为了克服以上问题，第二代GCN进行了针对性的改进：用k阶多项式来近似卷积核：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?g_%7B%5Ctheta%7D%28%5CLambda%20%29%20%5Capprox%20%5Csum_%7Bk%3D0%7D%5EK%5Ctheta%20_%7Bk%7D%5CLambda%20%5Ek)，将其代入到![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?Udiag%5B%5Chat%7Bh%7D%28%5Clambda%20_1%29%2C...%2C%5Chat%7Bh%7D%28%5Clambda%20_N%29%20%5DU%5ETf)可以得到![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%28g_%5Ctheta%20*%20x%29_G%20%5Capprox%20U%5Csum_%7Bk%3D0%7D%5EK%5Ctheta%20_%7Bk%7D%5CLambda%20%5EkU%5ETx%20%3D%20%5Csum_%7Bk%3D0%7D%5EK%5Ctheta%20_%7Bk%7D%28U%5CLambda%20%5EkU%5ET%29x%20%3D%20%5Csum_%7Bk%3D0%7D%5EK%5Ctheta%20_%7Bk%7D%28U%5CLambda%20U%5ET%29%5Ekx%20%3D%20%5Csum_%7Bk%3D0%7D%5EK%5Ctheta%20_%7Bk%7DL%5Ekx)

  所以第二代GCN的卷积公式是：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Csigma%20%28%5Csum_%7Bk%3D0%7D%5EK%5Ctheta%20_%7Bk%7DL%5Ekx%29)

  第二代GCN直接对拉普拉斯矩阵进行变换，不再需要特征分解这一耗时大户。

* ChebNet在第二代GCN的基础上用ChebShev多项式展开对卷积核进行近似，即令![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bcases%7D%20g_%20%5Ctheta%20%28%5CLambda%20%29%20%5Capprox%20%5Csum_%7Bk%3D0%7D%5E%7BK-1%7D%5Ctheta%20_%7Bk%7DT_k%28%5Chat%20%5CLambda%29%5C%5C%20%5Chat%20%5CLambda%20%3D%20%5Cfrac%7B2%7D%7B%5Clambda%20_%7Bmax%7D%7D%5CLambda%20-%20I_N%20%5Cend%7Bcases%7D)

  切比雪夫多项式的递归定义：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bcases%7D%20T_0%28x%29%20%3D%201%5C%5C%20T_1%28x%29%20%3D%20x%5C%5C%20T_%7Bn&plus;1%7D%28x%29%20%3D%202xT_n%28x%29%20-%20T_%7Bn-1%7D%28x%29%20%5Cend%7Bcases%7D)

  这样有两个好处：

  1. 卷积核的参数从原先一代GCN中的n个减少到k个，从原先的全局卷积变为现在的局部卷积，即将距离中心节点k-hop的节点作为邻居节点。

  2. 通过切比雪夫多项式的迭代定义降低了计算复杂度。

  因此切比雪夫图卷积公式变为：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bcases%7D%20y%20%3D%20%5Csigma%20%28%5Csum_%7Bk%3D0%7D%5EK%5Ctheta%20_%7Bk%7DT_k%28%5Chat%20L%29x%29%5C%5C%20%5Chat%20L%20%3D%20%5Cfrac%7B2%7D%7B%5Clambda%20_%7Bmax%7D%7DL%20-%20I_N%20%5Cend%7Bcases%7D)

**对上述推导过程不清楚的人可以参考我的博客**：https://www.jianshu.com/p/35212baf6671

教程完整代码链接：https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_chebnet.py

论文地址：https://arxiv.org/pdf/1606.09375.pdf

### 教程目录

***

* 开发环境
* ChebNet的实现
* 模型构建
* ChebNet训练
* ChebNet评估

### 开发环境

***

* 操作系统: Windows / Linux / Mac OS
* Python 版本: >= 3.5
* 依赖包:
  - tf_geometric（一个基于Tensorflow的GNN库）

根据你的环境（是否已安装TensorFlow、是否需要GPU）从下面选择一条安装命令即可一键安装所有Python依赖:



```python
	pip install -U tf_geometric # 这会使用你自带的TensorFlow，注意你需要tensorflow/tensorflow-gpu >= 1.14.0 or >= 2.0.0b1

	pip install -U tf_geometric[tf1-cpu] # 这会自动安装TensorFlow 1.x CPU版

	pip install -U tf_geometric[tf1-gpu] # 这会自动安装TensorFlow 1.x GPU版

	pip install -U tf_geometric[tf2-cpu] # 这会自动安装TensorFlow 2.x CPU版

	pip install -U tf_geometric[tf2-gpu] # 这会自动安装TensorFlow 2.x GPU版
	

```

教程使用的核心库是tf_geometric，一个基于TensorFlow的GNN库。tf_geometric的详细教程可以在其Github主页上查询：

* https://github.com/CrawlScript/tf_geometric

### ChebNet的实现

***

对图的邻接矩阵进行归一化处理得到拉普拉斯矩阵（归一化的方式有![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bcases%7D%20L%20%3D%20D%20-A%5C%5C%20L%5E%7Bsym%7D%20%3D%20D%5E%7B-1/2%7DLD%5E%7B-1/2%7D%5C%5C%20L%5E%7Brw%7D%20%3D%20D%5E%7B-1%7DL%20%5Cend%7Bcases%7D)），以及根据得到的归一化的拉普拉斯矩阵计算![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Chat%20L%20%3D%20%5Cfrac%7B2%7D%7B%5Clambda%20_%7Bmax%7D%7DL%20-%20I_N)。chebnet_norm_edge的具体实现请看[完整代码](https://github.com/CrawlScript/tf_geometric/blob/master/tf_geometric/nn/conv/chebnet.py)

```python
num_nodes = x.shape[0]
norm_edge_index, norm_edge_weight = chebnet_norm_edge(edge_index, num_nodes, edge_weight, lambda_max, normalization_type=normalization_type)                                            
```

利用切比雪夫多项式的迭代定义递推计算高阶项（节省了大量运算），最后输出模型结果，即多项式和![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Csigma%28%5Csum_%7Bk%3D0%7D%5EK%5Ctheta_kT_k%28%5Chat%20L%29x%29)计算loss或者评估模型效果。

```python
	T0_x = x
    T1_x = x
    out = tf.matmul(T0_x, kernel[0])

    if K > 1:
        T1_x = aggregate_neighbors(x, norm_edge_index, norm_edge_weight, gcn_mapper, sum_reducer, identity_updater)
        out += tf.matmul(T1_x, kernel[1])

    for i in range(2, K):
        T2_x = aggregate_neighbors(T1_x, norm_edge_index, norm_edge_weight, gcn_mapper, sum_reducer, identity_updater)  ##L^T_{k-1}(L^)
        T2_x = 2.0 * T2_x - T0_x
        out += tf.matmul(T2_x, kernel[i])

        T0_x, T1_x = T1_x, T2_x

    if bias is not None:
        out += bias

    if activation is not None:
        out += activation(out)

    return out
```

### 模型构建

***

* 导入相关库

  本教程使用的核心库是[tf_geometric](https://github.com/CrawlScript/tf_geometric)，我们用它来进行图数据导入、图数据预处理及图神经网络构建。ChebNet的具体实现已经在上面详细介绍，LaplacianMaxEigenvalue用来获取L阿普拉斯矩阵的最大特征值。另外我们后面会使用keras.metrics.Accuracy评估模型性能。

  ~~~python
  import os
  
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  import tensorflow as tf
  import numpy as np
  from tensorflow import keras
  from tf_geometric.layers.conv.chebnet import chebNet
  from tf_geometric.datasets.cora import CoraDataset
  from tf_geometric.utils.graph_utils import LaplacianMaxEigenvalue
  from tqdm import tqdm
  
  ~~~

* 使用[tf_geometric](https://github.com/CrawlScript/tf_geometric)自带的图结构数据接口加载Cora数据集：

  ```python
  graph, (train_index, valid_index, test_index) = CoraDataset().load_data()
  ```

* 获取图拉普拉斯矩阵的最大特征值

  ```python
  graph_lambda_max = LaplacianMaxEigenvalue(graph.x, graph.edge_index, graph.edge_weight)
  ```

* 定义模型,引入keras.layers中的Dropout层随机关闭神经元缓解过拟合。由于Dropout层在训练和预测阶段的状态不同，为此，我们通过参数training来决定是否需要Dropout发挥作用。

  ```
  model = chebNet(64, K=3, lambda_max=graph_lambda_max()
  fc = tf.keras.Sequential([
      keras.layers.Dropout(0.5),
      keras.layers.Dense(num_classes)])
  
  
  def forward(graph, training=False):
      h = model([graph.x, graph.edge_index, graph.edge_weight])
      h = fc(h, training=training)
      return h
  ```

### ChebNet训练

***

模型的训练与其他基于Tensorflow框架的模型训练基本一致，主要步骤有定义优化器，计算误差与梯度，反向传播等。

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

best_test_acc = tmp_valid_acc = 0
for step in tqdm(range(1, 101)):
    with tf.GradientTape() as tape:
        logits = forward(graph, training=True)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    valid_acc = evaluate(valid_index)
    test_acc = evaluate(test_index)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        tmp_valid_acc = valid_acc
    print("step = {}\tloss = {}\tvalid_acc = {}\tbest_test_acc = {}".format(step, loss, tmp_valid_acc, best_test_acc))
```

* 用交叉熵损失函数计算模型损失。注意在加载Cora数据集的时候，返回值是整个图数据以及相应的train_mask,valid_mask,test_mask。TAGCN在训练的时候的输入时整个Graph，在计算损失的时候通过train_mask来计算模型在训练集上的迭代损失。因此，此时传入的mask_index是train_index。由于是多分类任务，需要将节点的标签转换为one-hot向量以便于模型输出的结果维度对应。由于图神经模型在小数据集上很容易就会疯狂拟合数据，所以这里用L2正则化缓解过拟合。

  ```python
  def compute_loss(logits, mask_index, vars):
      masked_logits = tf.gather(logits, mask_index)
      masked_labels = tf.gather(graph.y, mask_index)
      losses = tf.nn.softmax_cross_entropy_with_logits(
          logits=masked_logits,
          labels=tf.one_hot(masked_labels, depth=num_classes)
      )
  
      kernel_vals = [var for var in vars if "kernel" in var.name]
      l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
  
      return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4
  
  ```

### ChebNet评估

***
在评估模型性能的时候我们只需传入valid_mask或者test_mask，通过tf.gather函数就可以拿出验证集或测试集在模型上的预测结果与真实标签，用keras自带的keras.metrics.Accuracy计算准确率。
```python
def evaluate(mask):
    logits = forward(graph)
    logits = tf.nn.log_softmax(logits, axis=-1)
    masked_logits = tf.gather(logits, mask)
    masked_labels = tf.gather(graph.y, mask)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(masked_labels, y_pred)
    return accuracy_m.result().numpy()
```

### 运行结果

***

```
 0%|          | 0/100 [00:00<?, ?it/s]step = 1	loss = 1.9817407131195068	valid_acc = 0.7139999866485596	best_test_acc = 0.7089999914169312
  2%|▏         | 2/100 [00:01<00:55,  1.76it/s]step = 2	loss = 1.6069653034210205	valid_acc = 0.75	best_test_acc = 0.7409999966621399
step = 3	loss = 1.2625869512557983	valid_acc = 0.7720000147819519	best_test_acc = 0.7699999809265137
  4%|▍         | 4/100 [00:01<00:48,  1.98it/s]step = 4	loss = 0.9443040490150452	valid_acc = 0.7760000228881836	best_test_acc = 0.7749999761581421
  5%|▌         | 5/100 [00:02<00:46,  2.06it/s]step = 5	loss = 0.7023431062698364	valid_acc = 0.7760000228881836	best_test_acc = 0.7770000100135803
  ...
96	loss = 0.0799005851149559	valid_acc = 0.7940000295639038	best_test_acc = 0.8080000281333923
 96%|█████████▌| 96/100 [00:43<00:01,  2.31it/s]step = 97	loss = 0.0768655389547348	valid_acc = 0.7940000295639038	best_test_acc = 0.8080000281333923
 97%|█████████▋| 97/100 [00:43<00:01,  2.33it/s]step = 98	loss = 0.0834992527961731	valid_acc = 0.7940000295639038	best_test_acc = 0.8080000281333923
 99%|█████████▉| 99/100 [00:44<00:00,  2.34it/s]step = 99	loss = 0.07315651327371597	valid_acc = 0.7940000295639038	best_test_acc = 0.8080000281333923
100%|██████████| 100/100 [00:44<00:00,  2.23it/s]
step = 100	loss = 0.07698118686676025	valid_acc = 0.7940000295639038	best_test_acc = 0.8080000281333923
```

### 完整代码

教程中的完整代码链接：

- demo_chebnet.py:https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_chebnet.py

