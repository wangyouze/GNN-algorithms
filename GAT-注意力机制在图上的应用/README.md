### Tensorflow-GAT-Tutorial

***

本教程将详细讲解如何用Tensorflow构建Graph Attention Networks（GAT)模型在Cora数据集上进行节点分类任务。完整代码可以在Github中进行下载：https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gat.py

### GAT简介

***

GCN通过图的拉普拉斯矩阵来聚合邻居节点的特征信息，这种方式和图本身的结构紧密相关，这限制了GCN在训练时未见的图结构上的泛化能力。GAT利用注意力机制来对邻居节点特征加权求和，从而聚合邻域信息，GAT完全摆脱了图结构的束缚，是一种归纳式学习方式。

<div align=center>
	<img src="gat.png" width="">
</div>

GAT中的attention是self-attention，即Q(Query)，K(Key)，V(value)三个矩阵均来自统一输入。和所有的Attention机制一样，GAT的计算也分两步走：

1. 计算注意力系数。对于中心节点，我们需要逐个计算它与它的邻居节点之间的注意力系数：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?e_%7Bij%7D%20%3D%20a%28%5BWh_i%20%5Cleft%20%7C%20%5Cright%20%7CWh_j%5D%29%2Cj%20%5Cin%20N_i)。具体来说，首先我们要计算中心节点Q向量与其邻居节点K向量之间的点乘，然后为了防止其结果过大，会除以一个尺度 ![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7Bd_k%7D) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=d_k) 为一个query（key向量）的维度。再利用Softmax操作将其结果归一化为概率分布。该操作可以表示为 ![](https://latex.codecogs.com/gif.latex?%5Calpha%20_%7Bij%7D%20%3D%20softmax%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%20d_k%7D%29)。

2. 通过加权求和的方式聚合节点信息。根据每个邻居节点对中心节点归一化后的注意力系数，对邻居节点的特征V进行线性组合作为中心节点的特征表示：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?h_i%5E%7B%27%7D%20%3D%20%5Csigma%28%5Csum%20_%7Bj%5Cin%20N_i%7D%5Calpha_%7Bij%7DWh_j%29)

   由于使用了multi-head attention，我们将K个head下的节点表示进行拼接作为最终的节点表示：

![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?h_i%5E%7B%27%7D%20%3D%20%5Cleft%20%7C%20%5Cright%20%7C_%7Bk%3D1%7D%5EK%20%5Csigma%28%5Csum%20_%7Bj%5Cin%20N_i%7D%5Calpha_%7Bij%7D%5EkW%5Ekh_j%29)

我们可以看出GAT的节点信息更新过程中是逐点运算的，每一个中心节点只与它的邻居节点有关，参数a和W也只与节点特征相关，与图结构无关。改变图的结构只需要改变节点的邻居关系![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?N_i)重新计算，因此GAT适用于inductive任务。

GCN中节点特征的每一次更新都需要全图参与，学习到的参数也很大程度上与图结构有关，因此GCN在inductive任务上不给力了。

GNN引入Attention机制有三大好处：

1. 参数少。Attention模型的复杂度与GCN等相比，复杂度更小，参数也更少，对算力的要求也就更小。
2. 速度快。Attention更加有利于并行计算。Attention机制中的每一步计算不依赖与前面一部的计算结果，并行性良好。
3. 效果好。Attention机制不会弱化对于长距离的信息记忆。



教程代码下载链接：https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gat.py

论文下载地址链接：https://arxiv.org/pdf/1710.10903.pdf

### 教程目录

***

* 开发环境
* GAT的实现
* 模型构建
* TAGCN训练
* TAGCN评估 

### 开发环境
***
* 操作系统: Windows / Linux / Mac OS
* Python 版本: >= 3.5
* 依赖包:
* tf_geometric（一个基于Tensorflow的GNN库）
	根据你的环境（是否已安装TensorFlow、是否需要GPU）从下面选择一条安装命令即可一键安装所有Python依赖:
```
pip install -U tf_geometric # 这会使用你自带的TensorFlow，注意你需要tensorflow/tensorflow-gpu >= 1.14.0 or >= 2.0.0b1

	pip install -U tf_geometric[tf1-cpu] # 这会自动安装TensorFlow 1.x CPU版

	pip install -U tf_geometric[tf1-gpu] # 这会自动安装TensorFlow 1.x GPU版

	pip install -U tf_geometric[tf2-cpu] # 这会自动安装TensorFlow 2.x CPU版

	pip install -U tf_geometric[tf2-gpu] # 这会自动安装TensorFlow 2.x GPU版
```

教程使用的核心库是tf_geometric，一个基于TensorFlow的GNN库。tf_geometric的详细教程可以在其Github主页上查询：

- https://github.com/CrawlScript/tf_geometric

### GAT的实现

***

self.attention，计算中心节点与其邻居节点的注意力系数。首先添加自环。

```python
	num_nodes = x.shape[0]

    # self-attention
    edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes)
```
row为中心节点序列，col为一阶邻居节点序列

```python
    row, col = edge_index
```
将节点特征向量X通过不同的变换得到Q(Query)，K(Key)和V(value)向量。通过[tf.gather](![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\SGPicFaceTpBq\24208\3A2AC162.gif))得到中心节点的特征向量Q和相应的邻居节点的特征向量K。

```python
    Q = query_activation(x @ query_kernel + query_bias)
    Q = tf.gather(Q, row)
    
    K = key_activation(x @ key_kernel + key_bias)
    K = tf.gather(K, col)
    
    V = x @ kernel
```
由于是multi-head attention，所以Q，K，V也需要划分为num_heads，即每一个head都有自己相应的Q，K，V。最后将Q，K矩阵相乘（每一个中心节点的特征向量与其邻居节点的特征向量相乘）得到的attention_score，通过segmen_softmax进行归一化操作。

```python

    # xxxxx_ denotes the multi-head style stuff
    Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
    edge_index_ = tf.concat([edge_index + i * num_nodes for i in range(num_heads)], axis=1)
    
    att_score_ = tf.reduce_sum(Q_ * K_, axis=-1)
    normed_att_score_ = segment_softmax(att_score_, edge_index_[0], num_nodes * num_heads)
```
将归一化后的attention系数当做边的权重来对邻居节点进行加权求和操作，从而更新节点特征。由于是multi-head attention，所以将同一个节点在每一个attention下的节点特征拼接输出。
```python
h_ = aggregate_neighbors(
        V_, edge_index_, normed_att_score_,
        gcn_mapper,
        sum_reducer,
        identity_updater
    )

    h = tf.concat(tf.split(h_, num_heads, axis=0), axis=-1)

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h
```

### 模型构建

***

* 导入相关库

  本教程使用的核心库是[tf_geometric](https://github.com/CrawlScript/tf_geometric)，我们用它来进行图数据导入、图数据预处理及图神经网络构建。GAT的具体实现已经在上面详细介绍，另外我们后面会使用keras.metrics.Accuracy评估模型性能。
  
  ```python
  # coding=utf-8
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  import tf_geometric as tfg
  import tensorflow as tf
  from tensorflow import keras
  ```

* 使用[tf_geometric](https://github.com/CrawlScript/tf_geometric)自带的图结构数据接口加载Cora数据集：

  ```python
  graph, (train_index, valid_index, test_index) = CoraDataset().load_data()
  ```

* 定义图模型。我们构建两层GAT，即GAT只聚合2-hop的邻居特征，Dropout层用来缓解模型过拟合。

  ```python
  gat0 = tfg.layers.GAT(64, activation=tf.nn.relu, num_heads=8, drop_rate=drop_rate, attention_units=8)
  gat1 = tfg.layers.GAT(num_classes, drop_rate=0.6, attention_units=1)
  dropout = keras.layers.Dropout(drop_rate)
  
  def forward(graph, training=False):
      h = graph.x
      h = dropout(h, training=training)
      h = gat0([h, graph.edge_index], training=training)
      h = dropout(h, training=training)
      h = gat1([h, graph.edge_index], training=training)
      return h
  ```

### GAT训练
模型的训练与其他基于Tensorflow框架的模型训练基本一致，主要步骤有定义优化器，计算误差与梯度，反向传播等。
***

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

for step in range(2000):
    with tf.GradientTape() as tape:
        logits = forward(graph, training=True)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))
```

* 用交叉熵损失函数计算模型损失。注意在加载Cora数据集的时候，返回值是整个图数据以及相应的train_mask,valid_mask,test_mask。GAT在训练的时候的输入是整个Graph，在计算损失的时候通过train_mask来计算模型在训练集上的迭代损失。因此，此时传入的mask_index是train_index。由于是多分类任务，需要将节点的标签转换为one-hot向量以便于模型输出的结果维度对应。由于图神经模型在小数据集上很容易就会疯狂拟合数据，所以这里用L2正则化缓解过拟合。

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

### GAT评估

***

在评估模型性能的时候我们只需传入valid_mask或者test_mask，通过[tf.gather](https://www.tensorflow.org/api_docs/python/tf/gather)函数就可以拿出验证集或测试集在模型上的预测结果与真实标签，用keras自带的keras.metrics.Accuracy计算准确率。

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
step = 20	loss = 1.784507393836975	accuracy = 0.7839999794960022
step = 40	loss = 1.5089114904403687	accuracy = 0.800000011920929
step = 60	loss = 1.243167757987976	accuracy = 0.8140000104904175
...
step = 1120	loss = 0.8608425855636597	accuracy = 0.8130000233650208
step = 1140	loss = 0.8169388771057129	accuracy = 0.8019999861717224
step = 1160	loss = 0.7581816911697388	accuracy = 0.8019999861717224
step = 1180	loss = 0.8362383842468262	accuracy = 0.8009999990463257
```

### 完整代码

***

教程中完整代码链接：demo_gat.py:教程代码下载链接：https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gat.py