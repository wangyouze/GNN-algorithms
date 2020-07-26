### Tensorflow-GIN-Tutorial

***

How Powerful are Graph Neural Networks?相信有很多人会和我一样在边吃泡面边跑模型的时候会产生这样的疑惑。本教程将手把手教大家搭建基于Tensorflow的GIN模型，同时把该论文（ICLR 2019 best student paper)中关于GNNs为什么有效果的观点简要介绍。

### GIN简介

***

如GCN和GraphSAGE，都是通过迭代聚合一阶邻居信息来更新节点的特征表示，可以拆分为三个步骤：

1. Aggregate:聚合一阶邻居节点的特征。
2. Combine:将邻域特征与中心节点的特征融合，更新中心节点的特征。
3. 如果是图分类任务，需要把Graph中所有节点特征转换为Graph的特征表示。

上述过程与[Weisfeiler-Lehman(WL) test](http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf) 中的过程非常相似。WL_test是判断两个Graph结构是否相同的有效方法，主要通过迭代以下步骤来判断Graph的同构性：

 （初始化：将节点的id作为自身的标签。）

1. 聚合：将邻居节点和自身的标签进行聚合。

2. 更新节点标签：使用Hash表将节点聚合标签映射作为节点的的新标签。

   WL_test迭代过程如下图所示：

   ![](WL_test.jpg)

   ​								(此图引用自知乎陈乐天的文章《Graph Neural Networks多强大？》阅读笔记 - 陈乐天的文章 - 知乎 https://zhuanlan.zhihu.com/p/62006729，如有侵权，请联系删除)

   上图a中的G图中节点1的邻居节点有节点4；节点2的邻居节点有节点3和节点5；节点3的邻居节点有节点2，节点4，节点5；节点4的邻居节点有节点1，节点3，节点5；节点5的邻居节点有节点2，节点3，节点4。步骤1聚合邻居节点和自身标签后的结果就是b图中的G。然后用Hash将聚合后的结果映射为一个新的标签，进行标签压缩，如图c。用压缩后的标签来替代之前的聚合结果，进行标签更新，如图d，G‘同理。

   对于Graph的特征表示，WL_test方法用迭代前后图中节点标签的个数作为Graph的表示特征，如图e所示。

从上图我们可以看出WL_test的迭代过程确实和GNN的聚合过程非常相似，并且作者也证明了WL_test是图神经网络聚合领域信息能力的上限。

作者提出如果GNN中的Aggregate,Combine和Readout函数是单射，则GNN可以达到上限，和WL_test一样。最后推导出基于MLP+SUM 的GIN模型：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?h%5E%7B%28k%29%7D_v%20%3D%20MLP%5E%7B%28k%29%7D%28%281&plus;%5Cepsilon%20%5E%7B%28k%29%7D%29%5Ccdot%20h%5E%7B%28k-1%29%7D_v%20&plus;%20%5Csum%20_%7Bu%20%5Cin%20N%28u%29%7Dh%5E%7B%28k-1%29%7D_v%29)

对于每轮迭代产生的节点特征求和，然后拼接作为Graph的特征表示：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?h_G%20%3D%20CONCAT%28sum%28%28h_v%5E%7B%28k%29%7D%7Cv%5Cin%20G%29%29%7Ck%3D0%2C1%2C...%2CK%29)

完整代码下载地址：https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gin.py

论文下载地址：https://arxiv.org/pdf/1810.00826.pdf

### 教程目录

***

* 开发环境
* GIN的实现
* 模型构建
* GIN训练
* GIN评估

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

### GIN的实现

***

gcn_norm_edge对图的邻接矩阵进行对称归一化处理：![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?D%5E%7B-0.5%7D%28I&plus;A%29D%5E%7B-0.5%7D)

```python
updated_edge_index, normed_edge_weight = gin_norm_edge(edge_index, x.shape[0], edge_weight, improved, cache)
```

对节点特征进行转换，聚合一阶邻域信息

```python
	x = x @ kernel
    h = aggregate_neighbors(
        x, updated_edge_index, normed_edge_weight,
        identity_mapper,
        sum_reducer,
        identity_updater
    )
```

combine设置为![[å¬å¼]](https://www.zhihu.com/equation?tex=1%2B%5Cepsilon)，更新中心节点表示

```python
 h = x * (1 + eps) + h
```

MLP拟合近似拟合单射函数

```python
if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h
```

### 模型构建

***

* 导入相关库

  本教程使用的核心库是[tf_geometric](https://github.com/CrawlScript/tf_geometric)，我们用它来进行图数据导入、图数据预处理及图神经网络构建。GIN的具体实现已经在上面详细介绍，另外我们后面会使用keras.metrics.Accuracy评估模型性能。

  ```python
  # coding=utf-8
  import os
  import tensorflow as tf
  import numpy as np
  from tensorflow import keras
  from sklearn.model_selection import train_test_split
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  ```

* 第一次加载COLLAB数据集，预计需要花费5分钟时间。数据集第一次被预处理之后，tf_geometric会自动保存预处理的结果，以便下一次调用。对于一个TU dataset会包含节点标签，节点属性等，每个graph的处理结果会被以字典形式保存，多个图的预处理结果以list的形式返回。

  ```python
  graph_dicts = tfg.datasets.TUDataset("COLLAB").load_data()
  ```
  
* 用数据构建Graph Object，即图模型输入的三要素：节点特征，边信息以及标签。

  ```python
  ef construct_graph(graph_dict):
      return tfg.Graph(
          x=create_fake_node_features(graph_dict["num_nodes"]),
          edge_index=graph_dict["edge_index"],
          y=graph_dict["graph_label"]  # graph_dict["graph_label"] is a list with one int element
      )
  
  graphs = [construct_graph(graph_dict) for graph_dict in graph_dicts]
  ```

  

* 定义模型，我们的模型有两层GIN和Pooling层组成，mlp作为分类器根据GIN输出的图特征表示进行图分类。

  ```python
  gin0 = tfg.layers.GIN(100, activation=tf.nn.relu)
  gin1 = tfg.layers.GIN(100, activation=tf.nn.relu)
  mlp = keras.Sequential([
      keras.layers.Dense(50),
      keras.layers.Dropout(drop_rate),
      keras.layers.Dense(num_classes)
  ])
  # dense = keras.layers.Dense(num_classes)
  
  
  def forward(batch_graph, training=False, pooling="sum"):
      # GCN Encoder
      h = gin0([batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight])
      h = gin1([h, batch_graph.edge_index, batch_graph.edge_weight])
  
      # Pooling
      if pooling == "mean":
          h = tfg.nn.mean_pool(h, batch_graph.node_graph_index)
      elif pooling == "sum":
          h = tfg.nn.mean_pool(h, batch_graph.node_graph_index)
      elif pooling == "max":
          h = tfg.nn.max_pool(h, batch_graph.node_graph_index)
      elif pooling == "min":
          h = tfg.nn.min_pool(h, batch_graph.node_graph_index)
  
      # Predict Graph Labels
      h = mlp(h, training=training)
      return h
  ```

### GIN训练

***
* 数据集划分

  ```python
train_graphs, test_graphs = train_test_split(graphs, test_size=0.1)
  ```
* 计算标签种类

  ```python
  num_classes = np.max([graph.y[0] for graph in graphs]) + 1
  ```
* 模型的训练与其他基于Tensorflow框架的模型训练基本一致，主要步骤有定义优化器，计算误差与梯度，反向传播等。我们将训练集中的Graphs以batch的形式输入模型进行训练，对于Graphs的划分可以调用我们tf_geometric中的函数create_graph_generator。
  
  ```python
  for step in range(20000):
      train_batch_graph = next(train_batch_generator)
      with tf.GradientTape() as tape:
          logits = forward(train_batch_graph, training=True)
          losses = tf.nn.softmax_cross_entropy_with_logits(
              logits=logits,
              labels=tf.one_hot(train_batch_graph.y, depth=num_classes)
          )
  
          kernel_vals = [var for var in tape.watched_variables() if "kernel" in var.name]
          l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
  
          loss = tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4
  
      vars = tape.watched_variables()
      grads = tape.gradient(loss, vars)
      optimizer.apply_gradients(zip(grads, vars))
  
      if step % 20 == 0:
          accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))
  ```
  
  

### GIN评估

***

在评估模型性能的时候我们将测试集中的图以batch的形式输入到我们的模型之中，用keras自带的keras.metrics.Accuracy计算准确率。

```
def evaluate():
    accuracy_m = keras.metrics.Accuracy()

    for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        logits = forward(test_batch_graph)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(test_batch_graph.y, preds)

    return accuracy_m.result().numpy()
```

### 运行结果

***

