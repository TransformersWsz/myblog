---
title: Uplift Model离线评估指标
mathjax: true
toc: true
date: 2024-09-16 16:28:35
updated: 2024-09-16 16:28:35
categories:
- Marketing
tags:
- AUUC
- Uplift Curve
- Qini coefficient
- Qini Curve
---
uplift建模难点在于无法获得个体的ground truth，因为它是反事实的。只能通过构造treatment和control两组镜像人群，对比两组人群的转化增量，来实现模型性能的评估。

<!--more-->

## Uplift Curve

#### 计算公式
$$
f(k)=\left(\frac{Y_k^T}{N_k^T}-\frac{Y_k^C}{N_k^C}\right)\left(N_k^T+N_k^C\right)
$$

具体计算步骤如下：
1. 模型对样本集预测，然后将样本按照预测得到的uplift value进行降序排序
2. 取topK个样本，计算得到 $f(k)$ 。以 $k$ 为横轴，$f(k)$ 为纵轴，画出Uplift Curve
   - $Y_k^T$ 表示topK个样本中， treatment组有转化的样本数，$Y_k^C$同理
   - $N_k^T$ 表示topK个样本中， treatment组的总样本数，$N_k^C$同理
3. Uplift Curve下的面积即是AUUC，AUUC越大，表示模型性能越好

#### 代码实践
```python
def uplift_curve(y_true, uplift, treatment):
    """Compute Uplift curve.

    For computing the area under the Uplift Curve, see :func:`.uplift_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.

        :func:`.perfect_uplift_curve`: Compute the perfect Uplift curve.

        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.

        :func:`.qini_curve`: Compute Qini curve.

    References:
        Devriendt, F., Guns, T., & Verbeke, W. (2020). Learning to rank for uplift modeling. ArXiv, abs/2002.05897.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
    y_true, uplift, treatment = y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]

    num_trmnt = stable_cumsum(treatment)[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    num_all = threshold_indices + 1

    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    curve_values = (np.divide(y_trmnt, num_trmnt, out=np.zeros_like(y_trmnt), where=num_trmnt != 0) -
                    np.divide(y_ctrl, num_ctrl, out=np.zeros_like(y_ctrl), where=num_ctrl != 0)) * num_all

    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values
```


## Qini Curve
当treatment组和control组的样本数量（在topK样本里）相差比较大的时候，Uplift Curve的计算会存在问题。因此Qini引入缩放因子来减少样本量级差异所带来的影响。

#### 计算公式
$$
g(k)=Y_k^T- Y_k^C \times \frac{N_k^T}{N_k^C}
$$

具体计算步骤如下：
1. 模型对样本集预测，然后将样本按照预测得到的uplift value进行降序排序
2. 取topK个样本，计算得到 $g(k)$ 。以 $k$ 为横轴，$g(k)$ 为纵轴，画出Qini Curve
   - $Y_k^T$ 表示topK个样本中， treatment组有转化的样本数，$Y_k^C$同理
   - $N_k^T$ 表示topK个样本中， treatment组的总样本数，$N_k^C$同理
3. Qini Curve下的面积即是Qini coefficient，Qini coefficient越大，表示模型性能越好

```python
def qini_curve(y_true, uplift, treatment):
    """Compute Qini curve.

    For computing the area under the Qini Curve, see :func:`.qini_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.uplift_curve`: Compute the area under the Qini curve.

        :func:`.perfect_qini_curve`: Compute the perfect Qini curve.

        :func:`.plot_qini_curves`: Plot Qini curves from predictions..

        :func:`.uplift_curve`: Compute Uplift curve.

    References:
        Nicholas J Radcliffe. (2007). Using control groups to target on predicted lift:
        Building and assessing uplift model. Direct Marketing Analytics Journal, (3):14–21, 2007.

        Devriendt, F., Guns, T., & Verbeke, W. (2020). Learning to rank for uplift modeling. ArXiv, abs/2002.05897.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]

    y_true = y_true[desc_score_indices]
    treatment = treatment[desc_score_indices]
    uplift = uplift[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]

    num_trmnt = stable_cumsum(treatment)[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    num_all = threshold_indices + 1

    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    curve_values = y_trmnt - y_ctrl * np.divide(num_trmnt, num_ctrl, out=np.zeros_like(num_trmnt), where=num_ctrl != 0)
    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values
```

___

## 参考
- [排序评估：AUUC、Qini coefficient](https://zhuanlan.zhihu.com/p/627342229)
- [Source code for sklift.metrics.metrics](https://www.uplift-modeling.com/en/latest/_modules/sklift/metrics/metrics.html#qini_auc_score)