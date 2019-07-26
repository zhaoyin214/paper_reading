# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

---

A. Kendall, Y. Gal, [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?][bnn_cv], NIPS (2017)

[bnn_cv]: https://arxiv.org/abs/1703.04977 "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"

---

## 摘要

随机不确定度（aleatoric uncertainty）：观测数据的内在噪声（noise inherent in the observations）

认知不确定度（epistemic uncertainty）：模型不确定度（uncertainty in the model），增加足够的数据能够消除该不确定度（uncertainty which can be explained away given enough data）

## 1 引言

## 2 相关工作

不确定度（uncertainty）通过模型参数或输出的概率分布描述（ormalised as probability distributions over either the model parameters, or model outputs）。

（1）认知不确定性建模：首先假设模型权值的先验分布（placing a prior distribution over a model’s weights）；然后给定训练数据，衡量权重的变化程度（trying to capture how much these weights vary given some data）。

（2）随机不确定度建模：模型输出分布。

### 2.1 深度贝叶斯学习中的认知不确定度（Epistemic Uncertainty in Bayesian Deep Learning）

贝叶斯神经网络（Bayesian neural network，BNN）

### 2.2 异方差随机不确定度（Heteroscedastic Aleatoric Uncertainty）


## 3 结合随机、认知不确定度（Combining Aleatoric and Epistemic Uncertainty in One Model）

### 3.1 结合认知、异方差随机不确定度（Combining Heteroscedastic Aleatoric Uncertainty and Epistemic Uncertainty）

### 3.2 异方差不确定度（Heteroscedastic Uncertainty as Learned Loss Attenuation）
