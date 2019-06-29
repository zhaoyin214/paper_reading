# Distance Metric Learning for Large Margin Nearest Neighbor Classification

---

Weinberger K Q . [Distance Metric Learning for Large Margin Nearest Neighbor Classification][dmnn][J]. NIPS, 2005.

[dmnn]: http://rogerioferis.com/VisualRecognitionAndSearch2014/material/papers/WeinbergerNIPS05.pdf "Distance Metric Learning for Large Margin Nearest Neighbor Classification"

---

## 摘要

度量（metric）学习的目标是使得$k$个最近邻样本总是属于同一类别，且不同类别样本之间的距离很大。

无需修改即可处理多分类（multiway）问题

## 1 引言

kNN分类器对未标记样本的分类规则为：训练集中与其最近的$k$个样本投票表决，因此kNN分类器的性能取决于其采用的距离度量（distance metric）。

通常kNN分类器采用欧氏距离（Euclidean distance metric）衡量样本相似度，然而，欧氏距离完全不考虑训练集数据的统计特性。

相关文献指出：从标注样本中学习得到的距离度量，能够显著提高kNN的分类能力。

本文给出一种针对kNN分类器的马氏距离度量（Mahanalobis distance metric）学习方案，该度量的优化目标是$k$个最近邻样本总是属于同一类别，且不同类别样本之间的距离不小于某一余量（large margin）。

大余量最近邻（large margin nearest neighbor，LMNN）分类

## 2 模型

训练集为$\left\{\mathbf{x}_{i}, y_{i} \right\}_{i = 1}^{n}$，其中样本$\mathbf{x}_{i} \in \mathcal{R}^{d}$，标签$y_{i}$为离散类别；

二进制矩阵$y_{ij} \in \{0, 1\}$表示标签$y_{i}$和$y_{j}$是否相同；

通过学习线性变换（linear transformation）$\mathbf{L}: \mathcal{R}^{d} \rightarrow \mathcal{R}^{d}$，并将其用于计算平方距离：

$$\mathcal{D}(\mathbf{x}_{i}, \mathbf{x}_{j}) = \| \mathbf{L}(\mathbf{x}_{i}, \mathbf{x}_{j}) \|^{2} \tag{1}$$

进而优化kNN分类。

### 目标邻居（target neighbors）

目标邻居：对于每个样本$\mathbf{x}_{i}$，指定$k$个与其标签相同的样本，则优化目标为最小化样本与其$k$个目标邻居的距离。

无先验知识（prior knowledge）条件下，目标邻居可指定为与$\mathbf{x}_{i}$标签相同且欧氏距离最小的$k$条样本。

$\eta_{ij} \in \{0, 1\}$表示样本$\mathbf{x}_{j}$是否为样本$\mathbf{x}_{i}$的目标邻居。

学习过程中，$\eta_{ij}$和$y_{ij}$均保持不变。

### 损失函数（cost function）

$$\epsilon(\mathbf{L}) = \sum_{ij} \eta_{ij} \| \mathbf{L}(\mathbf{x}_{i} - \mathbf{x}_{j}) \|^{2} +
c \sum_{ijl} \eta_{ij} (1 - y_{il})
{\left[ 1 + \| \mathbf{L}(\mathbf{x}_{i} - \mathbf{x}_{j}) \|^{2} -
\| \mathbf{L}(\mathbf{x}_{i} - \mathbf{x}_{l}) \|^{2} \right]}_{+}
\tag{2}$$

其中，${\left[z\right]}_{+} = \max(z, 0)$表示合页损失函数（hinge loss）。上式第一项用于惩罚与输入样本距离过大的目标邻居，第二项用于惩罚与输入样本类别不同且距离过小的样本。

### 大余量（large margin）

方程(2)中的第二项对应余量。只有当异类样本距离小于目标邻居距离加一个绝对单位距离时，合页损失才会被触发，因此损失函数倾向于使异类样本不侵入输入样本的目标邻域。

![](./img/lmnn_fig_1.png)

### 与支持向量机比较（Parallels with SVMs）


### 凸优化（Convex optimization）

方程(2)的最优化为半定规化（semidefinite programming，SDP）问题，SDP为凸函数（convex），因此方程(2)有全局最小值。将方程(1)重写为：

$$\mathcal{D}(\mathbf{x}_{i}, \mathbf{x}_{j}) =
(\mathbf{x}_{i} - \mathbf{x}_{j})^{\mathrm{T}} \mathbf{M} (\mathbf{x}_{i} - \mathbf{x}_{j})
 \tag{3}$$

其中，$\mathbf{M} = \mathbf{L}^{\mathrm{T}} \mathbf{L}$。合页损失可通过引入松弛变量（slack variables）$\xi_{ij}$（$\ \forall \left<i, j\right>, y_{ij} = 0$）模拟，则SDP描述为：

> Minimize $\sum_{ij} \eta_{ij} (\mathbf{x}_{i} - \mathbf{x}_{j})^{\mathrm{T}} \mathbf{M} (\mathbf{x}_{i} - \mathbf{x}_{j}) +
c \sum_{ijl} \eta_{ij} (1 - y_{il}) \xi_{il}$
> subject to
> (1) $(\mathbf{x}_{i} - \mathbf{x}_{l})^{\mathrm{T}} \mathbf{M} (\mathbf{x}_{i} - \mathbf{x}_{l}) -
(\mathbf{x}_{i} - \mathbf{x}_{j})^{\mathrm{T}} \mathbf{M} (\mathbf{x}_{i} - \mathbf{x}_{j}) \ge
1 - \xi_{il}$
> (2) $\xi_{il} \ge 0$
> (3) $\mathbf{M} \succeq 0$

约束(3)表示$\mathbf{M}$为半正定矩阵（positive semidefinite）。

由于大多数的标识样本对相距较远，故其距离不会触发合页损失，因此松弛变量矩阵$\{\xi_{ij}\}$是稀疏的，即主动约束（active constraints）数量稀少。

## 3 结果

实验数据集

| | Iris | Wine | Faces | Bal | Isolet | News | MNIST |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| examples | (train) | 106 | 126 | 280 | 445 | 6238 | 16000 | 60000 |
| examples | (test) | 44 | 52 | 120 | 90 | 1559 | 2828 | 10000 |
| classes | 3 | 3 | 40 | 3 | 26 | 20 | 10 |
| input | dimensions | 4 | 13 | 1178 | 4 | 617 | 30000 | 784 |
| features | after | PCA | 4 | 13 | 30 | 4 | 172 | 200 | 164 |
| constraints | 5278 | 7266 | 78828 | 76440 | 37 | Mil | 164 | Mil | 3.3 | Bil |
| active | constraints | 113 | 1396 | 7665 | 3099 | 45747 | 732359 | 243596 |
| CPU | time | (per | run) | 2s | 8s | 7s | 13s | 11m | 1.5h | 4h |
| runs | 100 | 100 | 100 | 100 | 1 | 10 | 1 |

![](./img/lmnn_fig_2.png)
