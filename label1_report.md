# Label 1（中出力周）场景生成结果分析报告

## 1. 分析目标

本报告聚焦 output_label=1（中出力周）的场景生成质量评估。

说明：
- 任务是“场景生成（generation）”，不是“点预测（forecasting）”。
- 因此重点看分布匹配、时序统计、跨变量协同关系和多样性，而不是 RMSE/MAE 这类逐点预测指标。

## 2. 数据背景与标签分布

真实数据标签样本量：
- label 0（低出力）：2 周
- label 1（中出力）：40 周
- label 2（高出力）：10 周

结论：
- label 1 样本量在三类中最多，是当前最可信的主分析对象。
- label 0/2 样本量偏少，评估波动更大，可在报告中说明“样本不足导致不稳定”。

## 3. 中出力周范围如何定义

### 3.1 现实可执行定义（基于当前数据）

由于当前数据仅给出 output_label，没有附带阈值生成脚本，建议在报告中采用“经验定义”：

- 以每周可再生出力均值（风+光，168h）作为定义指标：
  - real label 0 范围：0.1193 ~ 0.1442
  - real label 1 范围：0.1535 ~ 0.2973
  - real label 2 范围：0.3037 ~ 0.3764

可见 label 1 与 0/2 基本分离，可近似理解为：
- 中出力周约在可再生周均值 0.15 ~ 0.30 区间（归一化尺度）

### 3.2 若报告需要一句话定义

可写为：

“中出力周（label 1）定义为周内风光联合出力水平位于中等区间的周样本；在本数据集中可再生周均值经验范围约为 0.1535–0.2973（归一化后）。”

## 4. Label 1 生成质量指标解读

指标文件：
- [experiments/baseline/eval_label1/metrics.json](experiments/baseline/eval_label1/metrics.json)

### 4.1 分布匹配（越小越好 / 越高越好）

- load_w1_quantile = 0.0191
- wind_w1_quantile = 0.0338
- solar_w1_quantile = 0.0081

解释：
- 三者都在可接受区间，太阳能最好，风电相对偏差最大。

覆盖率：
- load_p1_p99_coverage = 0.9979
- wind_p1_p99_coverage = 0.9988
- solar_p1_p99_coverage = 0.9929

解释：
- 生成值基本落在真实分布主要区间内，几乎没有明显异常值外溢。

### 4.2 时序统计匹配（越小越好）

- 均值曲线误差：
  - load_mean_profile_mae = 0.0124
  - wind_mean_profile_mae = 0.0279
  - solar_mean_profile_mae = 0.0134
- 标准差曲线误差：
  - load_std_profile_mae = 0.0325
  - wind_std_profile_mae = 0.0350
  - solar_std_profile_mae = 0.0085

解释：
- 太阳能时序统计拟合较好。
- 风电仍是主要误差来源，提示风电动态模式仍可进一步改进。

### 4.3 跨变量协同关系（越小越好）

- cross_var_corr_frobenius = 0.5506

解释：
- 负荷-风-光之间的相关结构已学到主要趋势，但仍存在中等偏差。
- 在当前 52 周小样本条件下，该水平可视作“可用但非最优”。

### 4.4 多样性（接近 1 最好）

- diversity_real_pairdist_mean = 4.6149
- diversity_gen_pairdist_mean = 3.7364
- diversity_ratio_gen_over_real = 0.8096

解释：
- 生成样本多样性低于真实样本，存在一定模式收缩，但不是严重崩塌。
- 对场景应用而言“可用”，但建议继续提升到接近 0.9~1.0。

## 5. 结论（可直接写入报告）

建议结论文本：

“在中出力周（label 1）条件下，模型能够生成与真实样本在边缘分布、主要时序统计特征和跨变量相关关系上基本一致的场景，场景质量总体达到可用水平。当前主要不足在于风电维度拟合误差相对较高以及生成多样性略偏保守（diversity ratio = 0.81），后续可通过扩充样本与小范围调参进一步提升。”

## 6. 下一步建议

1. 论文/报告主文先聚焦 label 1：
- 理由：样本量最大（40 周），结论最稳健。

2. label 0/2 可作为补充：
- 强调样本不足（2 周、10 周）导致评估方差较大。

3. 下一阶段优先方向：
- 引入更大规模数据集（尤其补足低/高出力周样本）。
- 以“跨变量相关误差 + 多样性比值”作为首要优化目标。

## 7. 可视化引用（报告插图）

- [experiments/baseline/eval_label1/mean_std_profiles.png](experiments/baseline/eval_label1/mean_std_profiles.png)
- [experiments/baseline/eval_label1/marginal_distributions.png](experiments/baseline/eval_label1/marginal_distributions.png)
- [experiments/baseline/eval_label1/cross_var_correlations.png](experiments/baseline/eval_label1/cross_var_correlations.png)
