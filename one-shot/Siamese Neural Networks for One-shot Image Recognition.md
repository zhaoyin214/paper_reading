# Siamese Neural Networks for One-shot Image Recognition

---

Gregory Koch, R. Zemel, R. Salakhutdinov, [Siamese Neural Networks for One-shot Image Recognition][siamese_net], ICML (2015)

[siamese_net]: http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf "Siamese Neural Networks for One-shot Image Recognition"

---

## 摘要

## 1 引言

## 2 相关工作

## 3 深度孪生网络

### 3.1 模型

孪生卷积神经网络（siamese convolutional neural network）

<img src="./img/siamese_net_fig_4.png" width="800" />

输出层：计算孪生分支的导出距离测度（induced distance metric between each siamese twin），其激活为sigmoid。

$$\mathbf{p} = \sigma(\sum_{j} \alpha_{j} \left| \mathbf{h}_{1, L-1}^{(j)} - \mathbf{h}_{2, L-1}^{(j)} \right|)$$

### 3.2 学习

* 损失函数（loss function）

正则化交叉熵损失（regularized cross-entropy objective）

$$\mathcal{L}(x_{1}^{(i)}, x_{2}^{(i)}) =
\mathbf{y}(x_{1}^{(i)}, x_{2}^{(i)}) \log \mathbf{p}(x_{1}^{(i)}, x_{2}^{(i)}) +
(1 - \mathbf{y}(x_{1}^{(i)}, x_{2}^{(i)})) \log (1 - \mathbf{p}(x_{1}^{(i)}, x_{2}^{(i)})) +
\mathbf{\lambda}^{\mathrm{T}} \left| \mathbf{w} \right|^{2}$$


* 优化（optimization）

$$\mathbf{w}^{(T)}_{kj} (x_{1}^{(i)}, x_{2}^{(i)}) =
\mathbf{w}^{(T)}_{kj} + \Delta \mathbf{w}^{(T)}_{kj} (x_{1}^{(i)}, x_{2}^{(i)}) + 2\lambda_j|\mathbf{w}_{kj}|$$

$$\Delta \mathbf{w}^{(T)}_{kj} (x_{1}^{(i)}, x_{2}^{(i)}) = - \eta_j \nabla \mathbf{w}^{(T)}_{kj} + \mu_j \Delta \mathbf{w}^{(T - 1)}_{kj}$$

其中，$\nabla \mathbf{w}^{(T)}_{kj}$为偏导数（partial derivative）

* 权值初始化（weight initialization）

* 学习策略（learning schedule）

* 超参优化（hyperparameter optimization）

* 仿射变形（affine distortions）

## 4 实验

### 4.1 Omniglot数据集

### 4.2 验证

<img src="./img/siamese_net_table_1.png" width="400" />

<img src="./img/siamese_net_fig_7.png" width="400" />

### 4.3 单次学习

<img src="./img/siamese_net_table_2.png" width="400" />

### 4.4 MNIST单次测试

<img src="./img/siamese_net_table_3.png" width="400" />


## 5 结论

