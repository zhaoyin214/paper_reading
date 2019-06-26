# SSD: Single Shot MultiBox Detector

---

W. Liu, et al., [SSD: Single Shot MultiBox Detector][ssd], ECCV (2016).

[ssd]: https://arxiv.org/abs/1512.02325 "SSD: Single Shot MultiBox Detector"

---

## 摘要

SSD：将边界框的输出空间离散化为一组在所有特征图（feature map）的每个位置上纵横比和尺度不同默认框；推理阶段，

网络为每个默认框中每个对象类别的存在生成评分，并对该框进行调整以更好地匹配对象形状


## 1 引言
