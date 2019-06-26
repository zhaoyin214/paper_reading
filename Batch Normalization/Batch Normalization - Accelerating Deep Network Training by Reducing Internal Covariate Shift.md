# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

---

Ioffe S , Szegedy C . [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift][batch_norm][J]. 2015.

[batch_norm]: https://arxiv.org/abs/1502.03167v2 "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"

---

## 摘要（Abstract）

由于训练过程中各层输入的分布随前一层参数的变化而变化，使得深度神经网络的训练变得复杂，同时也导致学习率较低、初始参数需要仔细调整、使用饱和非线性函数训练模型异常困难。这一现象被称为内部协变漂移（internal covariate shift，ICS）。

本文将标准化（normalization）加入模型结构，训练过程中，对迷你批量（mini-batch）进行标准化，以消除ISC。引入批量标准化后，学习率（learning rate）可以适当增大、并且放宽初始参数、甚至无需参数丢弃（dropout）。引入批量标准化后，模型预测准确度（accuracy）达到原始模型准确度时，训练步数仅为原始模型的7%。

## 1 引言（Introduction）

随机梯度下降（stochastic gradient descent，SGD）通过最小化损失函数对参数$\Theta$进行优化：

$$\Theta = \arg \min_{\Theta} \frac{1}{N} \sum_{i = 1}^{N} l \left( \mathbf{x}_i, \Theta \right)$$

其中，$\left\{ \mathbf{x}_i: i = 1, 2, \cdots, N \right\}$为训练数据集。对于包含$m$个样本的迷你批量（mini-batch）SGD，损失函数关于$\Theta$的梯度近似为：

$$\frac{1}{m} \sum_{i = 1}^{m} \frac{ \partial l \left( \mathbf{x}_i, \Theta \right) }{\partial \Theta}$$

考虑一复合网络：

$$l = F_2 \left( F_1 \left( u, \Theta_1 \right), \Theta_2 \right)$$

将$F_1$的输出作为$F_2$的输入$\mathbf{x} = F_1 \left( u, \Theta_1 \right)$：

$$l = F_2 \left( \mathbf{x}, \Theta_2 \right)$$

利用梯度下降（梯度下降、随机梯度下降推导可参见[读书笔记 - 机器学习实战 - 5 逻辑回归](https://blog.csdn.net/zhaoyin214/article/details/87437205)），$\Theta_2$的更新过程为：

$$\Theta_2 \leftarrow \Theta_2 - \frac{\alpha}{m} \sum_{i=1}^{m} \frac{\partial F_2 \left( \mathbf{x}_i, \Theta_2 \right)}{\partial \Theta_2}$$

其中，$m$为批量尺寸、$\alpha$为学习率。上式与单一网络$F_2$的学习过程表达式完全一致。因此能够使单一网络训练更有效的输入分布特性（如训练和测试数分布相同）同样适用于子网络的训练。

**若能确保$\mathbf{x}$的分布在训练期间保持不变，则**

（1）*无须为补偿$\mathbf{x}$的分布变化而调整$\Theta_2$*；

（2）*优化器不易陷入饱和区（get stuck in the saturated regime），防止梯度消失（vanishing gradients）*；

本文用*批量标准化（batch normalization，BN）*减轻*内部协变漂移（Internal Covariate Shift，ICS）*，并提升深度网络训练速度。

**BN**:

（1）固定各层输入的均值和方差，减轻ICS；

（2）减轻梯度对参数规模（scale of the parameters ）及其初值的依赖；

（3）提高学习率；

（4）对模型正则化（regularizes the model），减少对Dropout的需求；

（5）通过阻止网络陷入饱和状态，激活函数可采用饱和非线性函数（saturating nonlinearities），如sigmoid。


## 2 减轻内部协变漂移（Towards Reducing Internal Covariate Shift）


**内部协变漂移（Internal Covariate Shift，ICS）**：训练过程中网络参数的变化引起的网络激活分布的变化。

*假设第$s$层网络参数发生变化导致其输出（即第$s+1$层的输入）的分布改变，则第$s+1$层为补偿其输入分布的改变需调敕其网络参数，进而导致其输出（即第$s+2$层的输入）的分布改变，因此ICS会被逐层放大。网络层数越深，ICS越严重。*

*网络的输入经过白化（whitened，即输入先通过线性变换进行标准化（零均值、单位方差）后，再进行解相关）处理后，网络训练收敛速度会明显加快。*

如果标准化操作直接作用在优化步骤中，则梯度下降过程可能会试图以更新标准化的方式更新参数，因此削弱了梯度下降的效果。

$\mathbf{x}$：输入向量，$\mathbf{\mathcal{X}}$：训练数据集对应的$\mathbf{x}$的集合，标准化表示为：

$$\hat{\mathbf{x}} = \text{Norm} \left( \mathbf{x}, \mathbf{\mathcal{X}} \right)$$

即$\hat{\mathbf{x}}$不仅与给定的训练样本$\mathbf{x}$有关，还与$\mathbf{\mathcal{X}}$中的所有样本有关。同时，若$\mathbf{x}$是某层的输出，则$\mathbf{\mathcal{X}}$中的每条样本也都与$\Theta$有关。误差反向传播（backpropagation）时，需要计算雅克比矩阵：

$$
\frac{\partial \mathrm{Norm} \left( \mathbf{x}, \mathbf{\mathcal{X}} \right)}{\partial \mathbf{x}},
\quad
\frac{\partial \mathrm{Norm} \left( \mathbf{x}, \mathbf{\mathcal{X}} \right)}{\partial \mathbf{\mathcal{X}}}
$$

上式中，若忽略第二项，会导致梯度爆炸（explosion）

该框架中，对输入进行白化处理，计算代价巨大：

（1）协方差矩阵（covariance matrices）要在整个训练集上计算，这与随机梯度下降（stochastic gradient descent，SGD）矛盾；

（2）最本质的问题在于$\mathrm{Norm} \left( \mathbf{x}, \mathbf{\mathcal{X}} \right)$的计算涉及$\mathbf{\mathcal{X}}$的奇异值分解（Singular Value Decomposition，SVD），而SVD不是$\mathbf{\mathcal{X}}$的连续函数，导致$\frac{\partial \mathrm{Norm} \left( \mathbf{x}, \mathbf{\mathcal{X}} \right)}{\partial \mathbf{\mathcal{X}}}$并非处处存在。

## 3 基于迷你批量统计的标准化（Normalization via Mini-Batch Statistics）

本文对标准化进行了两项化简：

（1）独立标准化每个标量特征

给定一个$d$维输入$\mathbf{x} = \left( x^{(1)}, x^{(2)}, \cdots, x^{(d)} \right)$，在每个维度上分别做标准化：

$$\hat{x}^{(k)} = \frac{x^{(k)} - \mathrm{E} \left[ x^{(k)} \right]}{\sqrt{ \mathrm{Var} \left[ x^{(k)} \right]}}$$

其中，期望和方差是在整个训练集上计算的。

对输入进行简单的标准化可能会改变“层”的表示（$f(x) \rightarrow f(\hat{x})$，如对sigmoid的输入进行标准化可能导致本应处于非线性区的$x$落入线性区）。因此，*引入网络的变换必须是恒等变换（the transformation inserted in the networkcan represent the identity transform）*。为实现这一目标，本文为每个激活（activation，指前一层的输出激活）$x^{(k)}$引入缩放参数$\gamma^{(k)}$和平移参数$\beta^{(k)}$，

$$y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$

这些参数同原始模型参数一起训练，用于恢复网络的表达能力。若令$\gamma^{(k)} = \sqrt{ \mathrm{Var} \left[ x^{(k)} \right]}, \beta^{(k)} = \mathrm{E} \left[ x^{(k)} \right]$，即可恢复原始激活（$\hat{x}^{(k)} \rightarrow x^{(k)}$）。

（2）迷你批量激活均值、方差估计

对于随机优化（stochastic optimization）来说，利用整个训练数据集做激活标准化并不现实。本文利用迷你批量对激活均值和方差进行估计（each mini-batch produces estimates of the mean and variance of each activation），使得用于标准化的统计量能够完全参与梯度反向传播（gradient backpropagation）。

考虑样本数为$m$的迷你批量$\mathcal{B}$，由于各激活是独立标准化，将各激活统一记为$x$：

$$\mathcal{B} = \left\{ x_1, x_2, \cdots, x_m \right\}$$

批量标准化变换（Batch Normalizing Transform）：先将$\mathcal{B}$标准化为$\hat{x}_i$，再经线性变换为$y_i$：

$$\mathrm{BN}_{\gamma, \beta}: x_i \rightarrow y_i, \quad i = 1, 2, \cdots, m$$

* 迷你批量BN变换算法（算法1）：

![](./img/batch_norm_alg_1.png)

> 输入：$x$为迷你批量$\mathcal{B}$中的样本，$\gamma$和$\beta$为待学习参数
>
> 输出：$y_i = \mathrm{BN}_{\gamma, \beta} \left( x_i \right)$
>
> mini-batch mean: $\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i = 1}^{m} x_i$
>
> mini-batch variance: $\sigma_{\mathcal{B}}^2 \leftarrow \frac{1}{m} \sum_{i = 1}^{m} \left( x_i - \mu_{\mathcal{B}} \right)^2$
>
> normalize: $\hat{x}_i \leftarrow \frac{\left( x_i - \mu_{\mathcal{B}} \right)}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$
>
> scale ans shift: $y_i \leftarrow \gamma \hat{x}_i + \beta \equiv \mathrm{BN}_{\gamma, \beta} \left( x_i \right)$

可以将BN变换加入网络用于处理任意激活。

**注意**：

$x^{(k)}$表示网络第$s$层的第$k$个激活（即输出），$y^{(k)}$为网络第$s+1$层的输入，BN就是在网络相邻两层之间插入如下运算：

$$x^{(k)} \xrightarrow{\text{mini-batch normalization}} \hat{x}^{(k)} \xrightarrow{\text{linear transform}} y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$

**BP（backpropagate）算法中BN连续性推导**：

迷你批量所有样本损失和记为$l_{\Sigma} = \sum_{i = 1}^{m} l \left( x_i, \Theta \right)$，单条样本的损失记为$l \left( x_i, \Theta \right)$（原文在此处符号有混淆，作者未对单条样本损失（引言中记为$l$）与迷你批量内各样本的损失和加以区分，原文在此处$l$应表示迷你批量所有样本损失和），根据链式法则（chain rule）

（1）
$$\begin{aligned}
y_i
= & \gamma \hat{x}_i + \beta \\
\downarrow & \\
\frac{\partial l_{\Sigma}}{\partial \hat{x}_i}
= & \frac{\partial l}{\partial \hat{x}_i}
= \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \hat{x}_i} = \frac{\partial l}{\partial y_i} \cdot \gamma
\end{aligned}$$

（2）
$$\begin{aligned}
\hat{x}_i
= & \frac{\left( x_i - \mu_{\mathcal{B}} \right)}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \\
\downarrow & \\
\frac{\partial l_{\Sigma}}{\partial \sigma_{\mathcal{B}}^2}
= & \sum_{i = 1}^{m} \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \sigma_{\mathcal{B}}^2} \\
= & \sum_{i = 1}^{m} \frac{\partial l}{\partial \hat{x}_i} \cdot \left( x_i - \mu_{\mathcal{B}} \right) \cdot \frac{-1}{2} \cdot \left( \sigma_{\mathcal{B}}^2 + \epsilon \right)^{\frac{-3}{2}} \\
\end{aligned}$$

（3）
$$\begin{aligned}
\hat{x}_i
= & \frac{\left( x_i - \mu_{\mathcal{B}} \right)}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \\
\sigma_{\mathcal{B}}^2
= &\frac{1}{m} \sum_{i = 1}^{m} \left( x_i - \mu_{\mathcal{B}} \right)^2 \\
\downarrow & \\
\frac{\partial l_{\Sigma}}{\partial \mu_{\mathcal{B}}}
= & \sum_{i = 1}^{m} \left( \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \mu_{\mathcal{B}}} \right) +
\frac{\partial l_{\Sigma}}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{\partial \sigma_{\mathcal{B}}^2}{\partial \mu_{\mathcal{B}}} \\
= & \left( \sum_{i = 1}^{m} \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{- 1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \right) +
\frac{\partial l_{\Sigma}}{\partial \sigma_{\mathcal{B}}^2} \cdot \sum_{i = 1}^{m} \frac{-2 \left( x_i - \mu_{\mathcal{B}} \right)}{m} \\
\end{aligned}$$

（4）
$$\begin{aligned}
\hat{x}_i
= & \frac{\left( x_i - \mu_{\mathcal{B}} \right)}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \\
\sigma_{\mathcal{B}}^2
= &\frac{1}{m} \sum_{i = 1}^{m} \left( x_i - \mu_{\mathcal{B}} \right)^2 \\
\mu_{\mathcal{B}}
= & \frac{1}{m} \sum_{i = 1}^{m} x_i \\
\downarrow & \\
\frac{\partial l_{\Sigma}}{\partial x_i}
= & \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} +
\frac{\partial l_{\Sigma}}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{\partial \sigma_{\mathcal{B}}^2}{\partial x_i} +
\frac{\partial l_{\Sigma}}{\partial \mu_{\mathcal{B}}} \cdot \frac{\partial \mu_{\mathcal{B}}}{\partial x_i} \\
= & \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} +
\frac{\partial l_{\Sigma}}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{2 \left( x_i - \mu_{\mathcal{B}} \right)}{\partial \mu_{\mathcal{B}}} +
\frac{\partial l_{\Sigma}}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{1}{m}
\end{aligned}$$

（5）
$$\begin{aligned}
y_i
= & \gamma \hat{x}_i + \beta \\
\downarrow & \\
\frac{\partial l_{\Sigma}}{\partial \gamma}
= & \sum_{i = 1}^{m} \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma} \\
= & \sum_{i = 1}^{m} \frac{\partial l}{\partial y_i} \cdot \hat{x}_i
\end{aligned}$$

（6）
$$\begin{aligned}
y_i
= & \gamma \hat{x}_i + \beta \\
\downarrow & \\
\frac{\partial l_{\Sigma}}{\partial \beta}
= & \sum_{i = 1}^{m} \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \beta} \\
= & \sum_{i = 1}^{m} \frac{\partial l}{\partial y_i}
\end{aligned}$$

PS：
$
\frac{\partial l_{\Sigma}}{\partial y_i} = \frac{\partial l}{\partial y_i}，
\frac{\partial l_{\Sigma}}{\partial \hat{x}_i} = \frac{\partial l}{\partial \hat{x}_i}
$

因此，*BN变换是可微的*。将学到的仿射变换（learned affine transform）应用到标准化后的激活（normalized activations）上，则BN变换能够表达恒等变换（identity transformation）并保留网络能力（network capacity）

### 3.1 BN网络训练及推理（Training and Inference with Batch-Normalized Networks）

经迷你批量标准化的激活能够让训练更有效。但在推理（inference）过程中，并不需要（neither necessary nor desirable）迷你批量标准化。网络训练完成后，计算标准化时可用总体统计量（population statistics）代替迷你批量统计量。

$$\hat{x} = \frac{x - \mathrm{E} \left[ x \right]}{\sqrt{\mathrm{Var} \left[ x \right] + \epsilon}}$$

总体统计量为迷你批量统计量的无偏估计：

$$\mathrm{E} \left[ x \right] = \mathrm{E}_{\mathcal{B}} \left[ \mu_{\mathcal{B}} \right]$$

$$\mathrm{Var} \left[ x \right] = \frac{m}{m - 1} \cdot \mathrm{E}_{\mathcal{B}} \left[ \sigma_{\mathcal{B}}^{2} \right]$$

* 训练BN网络（算法2）：

![](./img/batch_norm_alg_2.png)

### 3.2 BN卷积网络（Batch-Normalized Convolutional Networks）

考虑一非线性仿射变换：

$$z = g(Wu + b)$$

其中，$W$和$b$为模型的学习参数（learned parameters），$g( \cdot )$为非线性函数（nonliearity，如sigmoid、ReLU）

BN变换层的位置：

（1）在非线性变换之前，对$Wu + b$标准化

（2）在非线性变换之后，对输入$u$标准化，但是因为你可能是另一个非线性的输出,其distri bution的形状可能会改变在训练,并限制其第一和第二时刻不会消除合作变量转变。相反，W u + b更可能具有对称的、非稀疏的分布，即“更高斯-西安”(Hyv¨arinen & Oja, 2000);正态化很可能产生分布稳定的活化。

我们通过对x = wu + b进行标准化，在非线性之前加上BN变换。
$$z = g\left( \mathrm{BN}(Wu) \right)$$



引用：

[魏秀参](https://www.zhihu.com/question/38102762)
> “Internal Covariate Shift”的解释：
>
> 在统计机器学习中，有一个经典假设是“源空间（source domain）和目标空间（target domain）的数据分布（distribution）是一致的”。如果不一致，那么就出现了新的机器学习问题，如transfer learning/domain adaptation等。
>
> covariate shift就是分布不一致假设下的一个分支问题，它是指源空间和目标空间的条件概率是一致的，但是其边缘概率不同，即：$P_s \left( Y | X = x \right) = P_t \left( Y | X = x \right), \ \forall x \in \mathcal{X}$，但$P_s \left( X \right) \not= P_t \left( X \right)$。
>
> 对于神经网络的各层输出，由于经过了层内操作，其分布与该层输入的分布不同，而且这种差异会随着网络深度增大而增大；另一方面，各层输出所“指示”的样本标签（label）是不变的，因此符合covariate shift的定义。由于是对层间信号的分析，故将其命名为“internal”。
>
> 那么好，为什么前面我说Google将其复杂化了。其实如果严格按照解决covariate shift的路子来做的话，大概就是上“importance weight”（ref）之类的机器学习方法。可是这里Google仅仅说“通过mini-batch来规范化某些层/所有层的输入，从而可以固定每层输入信号的均值与方差”就可以解决问题。如果covariate shift可以用这么简单的方法解决，那前人对其的研究也真真是白做了。此外，试想，均值方差一致的分布就是同样的分布吗？当然不是。显然，ICS只是这个问题的“包装纸”嘛，仅仅是一种high-level demonstration。那BN到底是什么原理呢？说到底还是为了防止“梯度弥散”。关于梯度弥散，大家都知道一个简单的栗子：。在BN中，是通过将activation规范为均值和方差一致的手段使得原本会减小的activation的scale变大。可以说是一种更有效的local response normalization方法（见4.2.1节）。
>

[CNN中batch normalization应该放在什么位置](https://www.zhihu.com/question/45270958)
> 为什么是对Activation Function的输入进行BN（即$\mathrm{BN}(Wu+b)$），而非对Hidden Layer的每一个输入进行BN（$W \mathrm{BN}(u) + b$）。按照作者的解释，由于Hidden Layer的输入$u$是上一层非线性Activation Function的输出，在训练初期其分布还在剧烈改变，此时约束其一阶矩和二阶矩无法很好地缓解Covariate Shift；而$\mathrm{BN}(Wu+b)$的分布更接近 Gaussian Distribution，限制其一阶矩和二阶矩能使输入到Activation Function的值分布更加稳定
