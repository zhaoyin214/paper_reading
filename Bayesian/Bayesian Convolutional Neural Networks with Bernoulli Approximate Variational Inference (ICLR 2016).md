# Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference

---
Y. Gal, Z. Ghahramani, [Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference][bayes_cnn], ICLR (2016)

[bayes_cnn]: https://arxiv.org/abs/1506.02158 "Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference"
---

## 摘要

## 1 引言

## 2 背景

### 2.1 概率建模和变分推理（probabilistic modelling and variational inference）

给定训练数据，输入$\{ \mathrm{x}_1, \dots, \mathrm{x}_N \}$、输出（标签）$\{ \mathrm{y}_1, \dots, \mathrm{y}_N \}$

概率建模（probabilistic modelling）任务是估计一个函数$\mathrm{y} = \mathrm{f} (\mathrm{x})$，使其根据输入$\mathrm{x}$生成“可能”的相应输出$\mathrm{y}$（likely to have generated our outputs）。

*贝叶斯方法（Bayesian approach）*：

（1）在函数空间上假设先验分布（put some prior distribution over the space of functions）$p(\mathrm{f})$。

（2）定义似然（likelihood）$p(\mathrm{Y} | \mathrm{f}, \mathrm{X})$：给定函数，描述观测生成的过程（capture the process in which observations are generated given a specific function）

（3）计算函数空间在给定数据集上的后验分布（look for the posterior distribution over the space of functions given our dataset）$p(\mathrm{f} | \mathrm{X}, \mathrm{Y})$

对新样本点$\mathrm{x}^{\ast}$的预测：在所有可能的$\mathrm{f}$上积分，

$$p(\mathrm{y}^{\ast} | \mathrm{x}^{\ast}, \mathrm{X}, \mathrm{Y}) =
\int p(\mathrm{y}^{\ast} | \mathrm{f}^{\ast})
p(\mathrm{f}^{\ast} | \mathrm{x}^{\ast}, \mathrm{X}, \mathrm{Y})
\text{d} \mathrm{f}^{\ast} \tag{1}$$

■
窃以为方程（1）有误，$\mathrm{f}^{\ast}$由$\mathrm{X}$、$\mathrm{Y}$估计，而与$\mathrm{x}^{\ast}$无关；$\mathrm{y}^{\ast}$取决于$\mathrm{f}^{\ast}$、$\mathrm{x}^{\ast}$。故方程（1）应修改为：

$$p(\mathrm{y}^{\ast} | \mathrm{x}^{\ast}, \mathrm{X}, \mathrm{Y}) =
\int p(\mathrm{y}^{\ast} | \mathrm{f}^{\ast}, \mathrm{x}^{\ast})
p(\mathrm{f}^{\ast} | \mathrm{X}, \mathrm{Y})
\text{d} \mathrm{f}^{\ast}$$

■

$$p(\mathrm{y}^{\ast} | \mathrm{x}^{\ast}, \mathrm{X}, \mathrm{Y}) =
\int p(\mathrm{y}^{\ast} | \mathrm{f}^{\ast})
p(\mathrm{f}^{\ast} | \mathrm{x}^{\ast}, \omega)
p(\omega | \mathrm{X}, \mathrm{Y})
\text{d} \mathrm{f}^{\ast} \text{d} \omega$$
