# Guided Image Filtering

---

K. He, J. Sun, X. Tang, [Guided Image Filtering][guided_filter], TPAMI (2013)

[guided_filter]: https://ieeexplore.ieee.org/document/6319316 "Guided Image Filtering"

---

## 摘要

本文提出一种名为引导滤波器（guided filter）的显式图像滤波器。引导滤波器由局部线性模型推导而来，通过考虑导频图像的内容来计算滤波输出。

## 1 引言

## 2 相关工作

### 2.1 显式加权平均滤波器（Explicit Weighted-Average Filters）

### 2.1 隐式加权平均滤波器（Implicit Weighted-Average Filters）

## 3 引导滤波器（Guided Filter）

线性移变滤波（linear translation-variant filtering）的一般形式：给定引导图像（a guidance image）$I$和滤波输入图像（an filtering input image）$p$（$I$和$p$可以相同），输出图像（an output image）为$q$，输出图像像素$i$为$I$对$p$的加权平均：

$$q_{i} = \sum_{j} W_{ij}(I) p_{j} \tag {1}$$

其中，$i$、$j$为像素索引（pixel indexes）；核（filter kernel）$W_{ij}$为引导图像$I$的函数且与$p$无关。引导滤波器与$p$线性相关（this filter is linear with respect to $p$）。

联合双边滤波器（joint bilateral filter）是一种线性移变滤波器（Fig. 1），其双边滤波核（bilateral filtering kernel）$W^{\text{bf}}$定义为：

$$W_{ij}^{\text{bf}} =
\frac{1}{K_{ij}}
\exp \left( - \frac{\| \mathbf{x}_{i} - \mathbf{x}_{j} \|^{2}}{\sigma_{s}^{2}} \right)
\exp \left( - \frac{\| I_{i} - I_{j} \|^{2}}{\sigma_{r}^{2}} \right)
\tag {2}$$

其中，$\mathbf{x}$为像素坐标（pixel coordinate）；$K_{ij}$为归一化参数（a normalizing parameter），以确保$\sum_{j} W_{ij}^{\text{bf}} = 1$；$\sigma_{s}$和$\sigma_{r}$分别用于调整空间位置相似性和数值相似性的敏感度（the sensitivity of the spatial similarity and the range (intensity/color) similarity）；当$I = p$时，即为双边滤波器（bilateral filter）。

隐式加权平均滤波器：优化二次函数并求解线性方程（optimize a quadratic function and solve a linear system）：

$$\mathbf{A} \mathbf{q} = \mathbf{p} \tag {3}$$

其中，$\mathbf{q} = \{ q_{i} \}$和$\mathbf{p} = \{ p_{i} \}$分别为$N \times 1$向量；$\mathbf{A}$为仅与$I$相关的$N \times N$矩阵（only depends on $I$）。方程（3）的解为$\mathbf{q} = \mathbf{A}^{-1} \mathbf{p}$，$W_{ij} = (\mathbf{A}^{-1})_{ij}$（形式上与方程（1）相同）。

<img src="./img/guided_filter_fig_1.png" width="800" />

### 3.1 定义

引导滤波器（guided filter）：假设引导滤波器（the key assumption of the guided filter）为引导图像$I$和滤波输入图像$p$间的局部线性模型（a local linear model），即$q$为$I$在像素$k$为中心的窗口$w_{k}$内的线性变换（$q$ is a linear transform of $I$ in a window $w_{k}$ centered at the pixel $k$）：

$$q_{i} = a_{k} I_{i} + b_{k}, \ \forall i \in w_{k} \tag {4}$$

其中，$(a_{k}, b_{k})$为线性系数（linear coefficients），在$w_{k}$中为常量；$w_{k}$为方形窗口，半径为$r$。由于$\nabla q = a \nabla I$，局部线性模型能确保$q$和$I$中的边缘位置相同（$q$ has an edge only if $I$ has an edge）。

$$q_{i} = p_{i} - n_{i} \tag {5}$$

$$E(a_{k}, b_{k}) = \sum_{i \in w_{k}} \left( (a_{k} I_{i} + b_{k} - p_{i})^{2} + \epsilon a_{k}^{2} \right) \tag {6}$$

### 3.2 保边滤波（Edge-Preserving Filtering）

### 3.3 核（Filter Kernel）

### 3.4 梯度不变滤波（Gradient-Preserving Filtering）

### 3.5 彩色滤波（Extension to Color Filtering）

### 3.6 结构转移滤波（Structure-Transferring Filtering）

### 3.7 （Relation to Implicit Methods）

## 4 计算复杂度（Computation and Efficiency）

## 5 实验

## 6 结论

