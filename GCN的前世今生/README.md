### 前言

<span style="color:red;">***由于GitHub目前不支持LaTeX，导致下面部分公式无法正常显示，有兴趣的童鞋可以转战我的博客：https://www.jianshu.com/p/35212baf6671***</span>

Graph Convolutional Networks涉及到两个很重要的概念：graph和Convolution。传统的卷积方式在欧式数据空间中大展神威，但是在非欧式数据空间中却哑火，很重要的一个原因就是传统的卷积方式在非欧式的数据空间上无法保持“平移不变性”。为了能够将卷积推广到Graph等非欧式数据结构的拓扑图上，GCN横空出世。在深入理解GCN：$H^{(l+1)} = \hat{D} ^{-1/2}\hat{A} \hat{D} ^{-1/2}H^lW^l$的来龙去脉之前，我觉着我们有必要提前对以下概念有点印象：

- 卷积和傅里叶变换本身存在着密不可分的关系。数学上的定义是两个函数的卷积等于各自傅里叶变换后的乘积的逆傅里叶变换。此时卷积与傅里叶变换产生了联系。

- 传统的傅里叶变换可以通过类比推广到图上的傅里叶变换。此时傅里叶变换又与Graph产生了联系。

- 由于傅里叶充当了友谊的桥梁，此时卷积和Graph终于搭上了线。

论文链接[Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)

demo_gcn.py:https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gcn.py

* * *
### 拉普拉斯矩阵与GCN
1.拉普拉斯矩阵及其变体
给定一个节点数为$n$的简单图$G$，$D$是$G$的度矩阵，$A$是$G$的邻接矩阵，则$G$的拉普拉斯矩阵可以表示为$L = D - A$.$L$中的各个元素表示如下：
$$L_{i,j}:=
\begin{cases}
diag(v_i)& \text{i = j}\\
-1& \text{if $i \neq j$ and $v_i$ adjacent to $v_j$}\\
0& othewise
\end{cases}$$
![拉普拉斯矩阵.png](https://upload-images.jianshu.io/upload_images/23355443-85097cc462a24cc5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
拉普拉斯矩阵变体：

- 对称归一化的拉普拉斯矩阵：$L^{sys} = D^{-1/2}LD^{-1/2} = I - D^{-1/2}AD^{-1/2}$
$$L^{sys}_{i,j}:=
\begin{cases}
0& \text{i = j}\\
-1\over\sqrt{diag(v_i)diag(v_j)}& \text{if $i \neq j$ and $v_i$ adjacent to $v_j$}\\
0& othewise
\end{cases}$$
- 随机游走归一化的拉普拉斯矩阵：
$L^{rw} = D^{-1}L = I - D^{-1}A$
$$L^{rw}_{i,j}:=
\begin{cases}
0& \text{i = j}\\
-1\over\sqrt{diag(v_i)}& \text{if $i \neq j$ and $v_i$ adjacent to $v_j$}\\
0& othewise
\end{cases}$$
2.拉普拉斯矩阵的优良性质：
- 拉普拉斯矩阵是半正定对称矩阵
- 对称矩阵有$n$个线性无关的特征向量，$n$是Graph中节点的个数$\Rightarrow$拉普拉斯矩阵可以特征分解
- 半正定矩阵的特征值非负
- 对称矩阵的特征向量构成的矩阵为正交阵$\Rightarrow U^TU = E$
3.GCN为什么要用拉普拉斯矩阵
- 拉普拉斯矩阵可以谱分解（特征分解）GCN是从谱域的角度提取拓扑图的空间特征的。
- 拉普拉斯矩阵只在中心元素和一阶相邻元素处有非零元素，其他位置皆为0.
- 传统傅里叶变换公式中的基函数是拉普拉斯算子，借助拉普拉斯矩阵，通过类比可以推导出Graph上的傅里叶变换公式。
***
### 傅里叶变换与GCN
1.传统的傅里叶变换
$F(w) = F[f(t)] = \int_{}^{} f(t)e^{-iwt}dt$
为离散变量时，对离散变量求积分相当于求内积，即$F(f(t)) = <f(t),e^{-iwt}>$
这里的$e^{-iwt}$就是传说中似乎有点神秘的拉普拉斯算子的特征函数（拉普拉斯算子是欧式空间中的二阶微分算子，卸了妆之后的样子是$ \vec{\nabla}^2 f= \vec{\nabla}\cdot(\vec{\nabla}f)$）。
为何这样说呢？是因为从广义的特征方程定义看$AV = \lambda V$, $A$本身是一种变换，$V$是特征向量或者特征函数，$\lambda$是特征值。我们对基函数$e^{-iwt}$求二阶导，$\Delta e^{-iwt} = \frac{\vartheta ^2 }{\vartheta t^2}e^{-iwt}= -w^2e^{-iwt} = ke^{-iwt}$. 可以看出$e^{-iwt}$是变换$\Delta$的特征函数。

在Graph中，拉普拉斯矩阵$L$可以谱分解(特征分解)，其特征向量组成的矩阵是$U$，根据特征方程的定义我们可以得到$LU = \lambda U$ 。通过对比我们可以发现$L$相当于$\Delta$,$U$相当于$e^{-iwt}$。因此在Graph上的傅里叶变换可以写作$F[f(\lambda _k)] = \hat{f}(\lambda_k) = <f,U_k> = \sum_{i=1}^nf(i)*U_k(i)$.

从傅里叶变换的基本思想来看，对$f(t)$进行傅里叶变换的本质就是将$f(t)$转换为一组正交基下的坐标表示，进行线性变换，而坐标就是傅里叶变换的结果，下图中的$\hat{f}_1$就是$f$在第一个基上的投影分量的大小。
![傅里叶转换思想.png](https://upload-images.jianshu.io/upload_images/23355443-d0725adb3a19e6b4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
而这跟拉普拉斯矩阵特征分解的本质是一样的。所以我们可以很自然的类比出Graph上的傅里叶变换为$\hat{f}(\lambda _k) = \sum_{i=1}^nf(i)*U_k(i)$，$\hat{f}(\lambda _k)$是$f$在$U_k$这个基的投影分量。

我们通过矩阵乘法将Graph上的傅里叶变换推广到矩阵形式：
![傅里叶变换矩阵形式.png](https://upload-images.jianshu.io/upload_images/23355443-0a146794e951e457.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
$f(N)$是Graph上第$N$个节点的特征向量，可得Graph上的傅里叶变换形式：$\hat{f}(\lambda) = U^Tf$。
此处的$U^T$是Graph的拉普拉斯矩阵的特征向量组成的特征矩阵的转置，在拉普拉斯矩阵的优良性质中我们知道拉普拉斯矩阵的特征向量组成的矩阵为正交阵，即满足$UU^T = E$，所以Graph的逆傅里叶变换形式为$f = U\hat{f}(\lambda)$，矩阵形式如下：
![逆傅里叶变换矩阵形式.png](https://upload-images.jianshu.io/upload_images/23355443-6cbc917e6f58820f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

到此为止我们已经通过类比从传统的傅里叶变换推广到了Graph上的傅里叶变换。接下来我们就要借助傅里叶变换这个桥梁使得Convolution与Graph把酒言欢了。
***
### 卷积与GCN
在前言中我们了解了大名鼎鼎的卷积定理：函数卷积的傅里叶变换是其傅里叶变换的乘积，即对于$f(t)与h(t)$,两者的卷积是其傅里叶变换的逆变换：
$(f*h)_G = F^{-1}(\hat{f}(w)\hat{h}(w))$
我们把上一节中得到的Graph上的傅里叶变换公式代入得到：
$(f*h)_G = F^{-1}((U^Tf)\odot(U^Th)) = U((U^Tf)\odot(U^Th))$  $\odot$是Hamada积，表示逐点相乘。
我们一般将$f$看作输入的Graph的节点特征，将$h$视为可训练且参数共享的卷积核来提取拓扑图的空间特征。为了进一步看清楚卷积核$h$，我们将上式改写为：
$U((U^Tf)\odot (U^Th)) = U((U^Th)\odot (U^Tf)) = Udiag[\hat{h}(\lambda _1),...,\hat{h}(\lambda _N)  ]U^Tf$
也许有人对于上式的变换心存疑虑，证明其实很简单，有意者请看这位答主的解答[GCN中的等式证明 - 知乎](https://zhuanlan.zhihu.com/p/121090537)

至此，我们已经推导出来GCN的雏形。
***
### GCN的进阶之路
1.**第一代GCN**
卷积操作的核心是由可训练且参数共享的卷积核，所以第一代GCN是直接把上式中的$diag[\hat{h}(\lambda _1),...,\hat{h}(\lambda _N)  ]$中的对角线元素$\hat{h}(\lambda_n)$替换为参数$\theta$。先初始化赋值，然后通过反向传播误差来调整参数$\theta$。
所以第一代GCN就变成了酱个样子：
$y = \sigma (Udiag[\hat{h}(\lambda _1),...,\hat{h}(\lambda _N)  ]U^Tf) = \sigma (Ug_\theta U^Tx)$
$x$是Graph中每个节点特征的表示向量，$y$是每个节点经过GCN卷积之后的输出。Graph中的每个节点都要经过卷积核卷积来提取其相应的拓扑空间，然后经过激活函数$\sigma$传播到下一层。
第一代GCN的缺点也是显而易见的，主要有以下几点，

- 需要对拉普拉斯矩阵进行特征分解，每次前向传播的过程中都要计算矩阵乘法，当Graph规模较大时，时间复杂度为$O(n^2)$，非常耗时。
- 卷积核的个数为$n$，因此当Graph中节点的个数$n$很大时，节点特征更新缓慢。

2.**第二代GCN**
面对第一代GCN参数过多的缺点，第二代GCN进行了针对性的改进。由于Graph上的傅里叶变换是关于特征值的函数$F(\lambda_N)$，$g_\theta$也可写作$g_\theta(\Lambda)$，用k阶多项式对卷积核进行改进:
$g(\theta )(\Lambda ) \approx \sum_{k=0}^K\theta _{k}\Lambda ^k  $
将其代入到$ Udiag[\hat{h}(\lambda _1),...,\hat{h}(\lambda _N)  ]U^Tf$可以得到：
$(g_\theta * x)_G \approx  U\sum_{k=0}^K\theta _{k}\Lambda ^kU^Tx  =  \sum_{k=0}^K\theta _{k}(U\Lambda ^kU^T)x =  \sum_{k=0}^K\theta _{k}(U\Lambda U^T)^kx = \sum_{k=0}^K\theta _{k}L^kx$
所以第二代GCN是介个样子：
$y = \sigma (\sum_{k=0}^K\theta _{k}L^kx)$
可以看出二代GCN的最终化简结果不需要进行矩阵分解，直接对拉普拉斯矩阵进行变换。参数是$\theta_k$，k一般情况下远小于Graph中的节点的数量$n$，所以和第一代GCN相比，第二代GCN的参数量明显少于第一代GCN，减低了模型的复杂度。对于参数$\theta_k$，先对其进行初始化，然后根据误差反向传播来更新参数。但是人就需要计算$L^k$，时间复杂度为$O(n^3)$.
另外我们知道对于一个矩阵的k次方，我们可以得到与中心节点k-hop相连的节点，即$L^k$中的元素是否为0表示Graph中的一个结点经过k跳之后是否能够到达另外一个结点，这里的k其实表示的就是卷积核感受野的大小，通过将每个中心节点k-hop内的邻居节点聚合来更新中心节点的特征表示，而参数$\theta_k$就是第k-hop邻居的权重。

3. ** 用切比雪夫多项式展开近似图卷积核。**
在二代GCN的基础上用ChebShev多项式展开对图卷积核进行近似，即令$$
\begin{cases}
g_ {\theta}\approx \sum_{k=0}^{K-1}\theta _k T_k(\hat \Lambda )\\
\hat \Lambda = \frac{2}{\lambda _{max}}\Lambda -I_N
\end{cases}$$
切比雪夫多项式的递归定义为：$$
\begin{cases}
T_0(x) = 1\\
T_1(x) = x\\
T_{n+1}(x) = 2xT_n(x) -T_{n-1}(x) 
\end{cases}$$
用切比雪夫多项式近似图卷积核有两个好处：
  - 卷积核的参数从原先一代GCN中的n个减少到k个，从原先的全局卷积变为现在的局部卷积，即将距离中心节点k-hop的节点作为邻居节点。
  -  通过切比雪夫多项式的迭代定义降低了计算复杂度。
因此切比雪夫图卷积公式为：$$
\begin{cases}
y = \sigma(\sum_{k=0}^{K}\theta _k T_k(\hat L)x))\\
\hat L = \frac{2}{\lambda _{max}}L -I_N
\end{cases}$$
手把手教你构建基于Tensorflow的ChebNet模型教程：https://github.com/wangyouze/GNN-algorithms
4.** 呱呱坠地的GCN **
上面啰嗦那么多就是为了等待GCN的呱呱坠地。GCN是在ChebNet的基础上继续化简得到的。
ChebNet的卷积公式为：$$
\begin{cases}
y = \sigma(\sum_{k=0}^{K}\theta _k T_k(\hat L)x))\\
\hat L = \frac{2}{\lambda _{max}}L -I_N
\end{cases}$$
令K=1，即只使用一阶切比雪夫多项式。
此时，$$y=\sigma(\sum_{k=0}^{1}\theta _k T_k(\hat L)x)) = \sigma(\theta_0T_0(\hat L)x + \theta_1T_1(\hat L)x)$$ 由切比雪夫多项式的迭代定义我们知道$$T_0(x) = 1, T_1(x) = x$$.所以$$ \sigma(\theta_0T_0(\hat L)x + \theta_1T_1(\hat L)x) = \sigma(\theta_0x + \theta_1\hat Lx)$$
令$\lambda_{max}=2$，则$\hat L = L - I_N$ 
上式$ \sigma(\theta_0x + \theta_1\hat Lx) = \sigma(\theta_0x + \theta_1(L - I_N)x)$
又$ L$ 是对称归一化的拉普拉斯矩阵，即$L = D^{-1/2}(D-A)D^{-1/2}$
因此上式$ \sigma(\theta_0x + \theta_1(L - I_N)x $
$= \sigma(\theta_0x + \theta_1(D^{-1/2}(D-A)D^{-1/2} - I_N)x) $
$= \sigma(\theta_0x + \theta_1(I_N - D^{1/2}AD^{-1/2} - I_N)x)$
$= \sigma(\theta_0x + \theta_1(- D^{1/2}AD^{-1/2} )x)  $
再令$\theta = \theta _0 = -\theta_1$ 
$\sigma(\theta_0x + \theta_1(- D^{1/2}AD^{-1/2})x)  = \sigma(\theta(I_N + D^{-1/2}AD^{-1/2})x) $
如果我们令$\hat A = I_N +A$
则$\sigma(\theta(I_N + D^{-1/2}AD^{-1/2})x) = \sigma(\theta D^{-1/2}\hat A D^{-1/2}x)$
将其推广到矩阵形式则得到我们耳熟能详的GCN卷积公式：$H^{(l+1)} = \hat{D} ^{-1/2}\hat{A} \hat{D} ^{-1/2}H^lW^l$

**未完待续。**

***
### 谱域卷积VS空域卷积
1.在谱域图卷积中，我们对图的拉普拉斯矩阵进行特征分解。通过在傅里叶空间中进行特征分解有助于我们我们理解潜在的子图结构。ChebNet, GCN是使用谱域卷积的典型深度学习架构。

2.空域卷积作用在节点的邻域上，我们通过聚合距离中心节点k-hop邻居来得到节点的特征表示。空域卷积相比谱域卷积更加简单和高效。GraphSAGE和GAT 是空域卷积的典型代表。
***
参考文献
1.[https://www.zhihu.com/search?type=content&q=GCN](https://www.zhihu.com/search?type=content&q=GCN)
2.[http://xtf615.com/2019/02/24/gcn/](http://xtf615.com/2019/02/24/gcn/)
3.[https://blog.csdn.net/yyl424525/article/details/100058264](https://blog.csdn.net/yyl424525/article/details/100058264)