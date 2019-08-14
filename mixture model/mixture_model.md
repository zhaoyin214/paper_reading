# 混合模型

*混合模型（mixture models）*：如果定义了观测变量和隐含（latent）变量的联合概率分布，则观测变量的概率分布可以通过计算边缘概率得到。因此引入隐含变量，可使复杂的概率分布通过简单的概率分布组合（混合）计算。

## K均值算法

数据集：$\{ \mathbf{x}_{1}, \cdots, \mathbf{x}_{N} \}$，$\mathbf{x}_{n} \in \R^{D}$，$\R^{D}$表示$D$维欧氏空间（Euclidean space）

聚类（无监督）成$K$个类别

*K均值算法（K-Means）*：首先随机指定$K$个类别中心$\mathbf{\mu}_{k} \in \R^{D}$，按最小欧氏距离（Euclidean distance）将样本逐条分配到$K$个类别；然后迭代更新类别中心$\mathbf{\mu}_{k}$和样本类别。

更新类别中心：

$$\mu_{k} = \frac{\sum_{n}^{N} \delta_{n, k} \mathbf{x}_{n}}{\sum_{n}^{N} \delta_{n, k}},
\quad
\delta_{n, k} = \begin{cases}
1, & \text{if } \mathbf{x}_{n} \in \text{class } k \\
0, & \text{otherwise} \\
\end{cases} \tag{1}$$

其中，$\delta_{n, k}$指示样本$\mathbf{x}_{n}$是否属于类别$k$。

* K均值算法的损失函数（loss function）：

$$\mathcal{L} = \sum_{n = 1}^{N} \sum_{k = 1}^{K} \delta_{n, k} {\| \mathbf{x}_{n} - \mathbf{\mu}_{k} \|}_{2}^{2} \tag{2}$$

* K均值算法的参数估计：

$$\{ \hat{\mathbf{\mu}}_{k} \}_{k = 1, \cdots, K},
\{ \hat{\delta}_{n, k} \}_{k = 1, \cdots, K, n = 1, \cdots, N} =
\argmin_{\begin{matrix}\{ \mathbf{\mu} \}_{k = 1, \cdots, K} \\
\{ \delta_{n, k} \}_{k = 1, \cdots, K, n = 1, \cdots, N} \end{matrix}} \mathcal{L}$$

* 优化（EM算法）

1) 初始化：随机指定$K$个类别中心$\mathbf{\mu}_{k}$；

2) 固定类别中心，求方程(2)关于$\delta_{n, k}$的最小值。显然，当各样本点被标记为与其欧氏距离最小的中心所属类别时，方程(2)取值最小；

3) 固定各样本所属类别$\delta_{n, k}$，求方程(2)关于$\mathbf{\mu}_{k}$的最小值：

$$\frac{\partial}{\partial \mathbf{\mu}_{k}} \mathcal{L} =
\begin{bmatrix}
\frac{\partial}{\partial \mu_{k, 1}} \mathcal{L}
& \cdots &
\frac{\partial}{\partial \mu_{k, D}} \mathcal{L}
\end{bmatrix} = 0$$

即

$$\frac{\partial}{\partial \mu_{k, d}} \mathcal{L} = \sum_{n = 1}^{N} 2 \delta_{n, k} ( \mu_{k, d} - x_{n, d} ) = 0$$

则

$$\mu_{k, d} = \frac{\sum_{n = 1}^{N} \delta_{n, k} x_{n, d}}{\sum_{n = 1}^{N} \delta_{n, k}}$$

4) 重复2)、3)直至收敛。

步骤2)更新$\delta_{n, k}$过程对应EM算法中的E（期望）步骤；步骤3)更新$\mathbf{\mu}_{k}$过程对应EM算法中的M（最大化）步骤。

EM算法是一种贪心算法，其E步骤和M步骤均减小损失函数的值，使算法的收敛性得到保证。

## 高斯混合模型

*高斯混合模型（Gaussian mixture model，GMM）*
