# Effective Approaches to Attention-based Neural Machine Translation

---

M.T. Luong, H. Pham, C. D. Manning, [Effective Approaches to Attention-based Neural Machine Translation][att_nmt], EMNLP (2015)

[att_nmt]: https://arxiv.org/abs/1508.04025v3 "Effective Approaches to Attention-based Neural Machine Translation"

---


## 摘要

全局注意力（a global approach）：（attends to all source words）

局部注意力（a local approach）：（only looks at a subset of source words at a time）

## 1 引言

<eos>：序列截止符（end-of-sentence symbol）

<img src="./img/att_nmt_fig_1.png" width="400" />

## 2 神经网络机器翻译（Neural Machine Translation）

神经网络机器翻译（neural machine translation，NMT）：神经网络直接对将源序列$x_1, x_2, \cdots, x_n$翻译为目标序列$y_1, y_2, \cdots, y_m$的条件概率$p(y | x)$建模。（A neural machine translation system is a neural network that directly models the conditional probability $p(y | x)$ of translating a source sentence, $x_1, x_2, \cdots, x_n$, to a target sentence, $y_1, y_2, \cdots, y_m$）

NMT的基本形式为：

（1）编码器（encoder）：计算每个源序列的表示$s$（a representation $s$ for each source sentence）；
（2）译码器（decoder）：每次生成一个目标词（generates one target word at a time）、分解条件概率，

$$\log p(y | x) = \sum_{j = 1}^{m} \log p(y_{j} | y_{\lt j},s)$$

## 3 注意力机制模型（Attention-based Models）

注意力机制模型（attention-based models）：
* 全局（global）：注意所有源位置（“attention” is placed on all source positions）
* 局部（local）：只注意部分源位置（“attention” is placed on only a few source positions）

<img src="./img/att_nmt_fig_2.png" width="400" />

<img src="./img/att_nmt_fig_3.png" width="400" />

共同点：在解码阶段每个时间步（at each time step）$t$上，二者均以堆叠LSTM最高层（at the top layer of a stacking LSTM）的隐含状态$\mathrm{h}_t$为输入；其目的为推导上下文向量（a context vector）$\mathrm{c}_t$，使$\mathrm{c}_t$能够捕获相关源信息（captures relevant source-side information），帮助预测当前目标词（current target word）$y_t$。

（1）使用连接层（a concatenation layer）将给定的$\mathrm{h}_t$、$\mathrm{c}_t$合并，生成注意力隐含状态（an attentional hidden state），

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_c[\mathrm{c}_t; \mathrm{h}_t]) \tag{5}$$

（2）将$\tilde{\mathbf{h}}_t$输入softmax层，以生成预测分布，

$$p(y_t | y_{\lt t}, x) = \text{sofmax}(\mathbf{W}_s \tilde{\mathbf{h}}_t) \tag{6}$$

### 3.1 全局注意力（Global Attention）

计算上下文向量$\mathrm{c}_t$时，全局注意力机制考虑编码器的全部隐含状态。该模型中，通过将当前目标隐含状态（current target hidden state）$\mathbf{h}_t$与每个源隐含状态（each source hidden state）$\bar{\mathbf{h}}_s$进行比较，计算变长对齐向量（a variable-length alignment vector）$\mathbf{a}_{t}$，$\mathbf{a}_{t}$的长度等于源序列的时间步数（the number of time steps on the source side）。

$$\begin{aligned}
\mathbf{a}_{t}(s) = & \text{align}(\mathbf{h}_t, \bar{\mathbf{h}}_s) \\
 = & \frac{\exp (\text{score}(\mathbf{h}_t, \bar{\mathbf{h}}_s))}
 {\sum_{s^{\prime}}\exp (\text{score}(\mathbf{h}_t, \bar{\mathbf{h}}_s^{\prime}))}
 \end{aligned} \tag{7}$$

其中，$\text{score}$评分是基于内容的函数（a content-based function），本文给出3种形式：

$$\text{score}(\mathbf{h}_t, \bar{\mathbf{h}}_s) =
\begin{cases}
\mathbf{h}_t^{\text{T}} \bar{\mathbf{h}}_s & \text{dot} \\
\mathbf{h}_t^{\text{T}} \mathbf{W}_a \bar{\mathbf{h}}_s & \text{general} \\
\mathbf{v}_a^{\text{T}} \tanh(\mathbf{W}_a [\mathbf{h}_t; \bar{\mathbf{h}}_s]) & \text{concat} \\
\end{cases}$$



### 3.2 局部注意力（Local Attention）

### 3.3 Input-feeding Approach

<img src="./img/att_nmt_fig_4.png" width="400" />

## 4 实验

### 4.1 训练细节

### 4.2 英文-德文结果

### 4.3 德文-英文结果


## 5 分析

### 5.1 学习曲线


### 5.2 长序列翻议

### 5.3 注意力结构选择


### 5.4 对齐质量（Alignment Quality）


### 5.5 翻译实例


## 6 结论

## 致谢
