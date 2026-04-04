# 多变量协同 CGAN（WGAN-GP）场景生成项目

本项目基于原论文思路与现有单变量代码重构，目标是生成山东周级多变量场景：

- 负荷（168 小时）
- 风电（168 小时）
- 光伏（168 小时）

总特征维度为 504。条件标签仅使用 output_label（0: 低出力，1: 中出力，2: 高出力），暂不使用 season_label。

算法采用条件 WGAN-GP（Wasserstein GAN with Gradient Penalty），以提升小样本（52 周）训练稳定性，降低模式崩溃风险。

说明：实现逻辑参考 2018-IEEE-Model-Free-GAN-wind.pdf 中的 GAN 场景生成思想，并按当前数据集与新需求做了工程化改造（多变量 504 维、仅 output_label 条件、WGAN-GP 稳定训练、推理逆归一化）。

## 1. 环境配置（Conda）

1. 创建环境

	 conda create -n shandong_cgan python=3.10 -y

2. 激活环境

	 conda activate shandong_cgan

3. 安装依赖

	 pip install -r requirements.txt

## 2. 数据准备说明

默认训练数据文件：

- shandong_gan_ready_final.csv

数据列要求：

- 前 504 列：归一化后的特征（0-1）
	- 0-167: 负荷
	- 168-335: 风电
	- 336-503: 光伏
- 条件标签列：output_label（取值 0/1/2）
- season_label 将被忽略

训练脚本会拟合并保存 MinMaxScaler 到 shandong_scaler.pkl。
推理时将加载该文件进行逆归一化，把生成值恢复到物理量单位 MW。

## 3. 训练命令

基础训练示例：

python train.py \
	--data shandong_gan_ready_final.csv \
	--epochs 800 \
	--batch-size 16 \
	--noise-dim 128 \
	--n-critic 5 \
	--gp-lambda 10.0 \
	--lr 1e-4 \
	--save-dir outputs

### 3.1 在终端实时查看训练日志（推荐）

如果你希望在当前终端持续看到训练进度，不要把训练放到后台，直接前台运行：

python train.py \
	--data germany_data_final.csv \
	--epochs 800 \
	--batch-size 16 \
	--noise-dim 128 \
	--n-critic 5 \
	--gp-lambda 10.0 \
	--lr 1e-4 \
	--save-dir result \
	--scaler-path germany_scaler.pkl

如果你想一边看日志，一边把日志保存到文件：

mkdir -p logs
python train.py \
	--data germany_data_final.csv \
	--save-dir result \
	--scaler-path germany_scaler.pkl \
	2>&1 | tee logs/train_germany.log

如果你已经在后台训练了，可以这样查看最新输出：

tail -f logs/train_germany.log

说明：当前 train.py 每 20 个 epoch 打印一次日志（以及第 1 个 epoch）。

训练输出：

- 模型权重：outputs/<时间戳>/cgan_wgangp_504.pth
- 损失曲线数据：outputs/<时间戳>/loss_history.csv
- 损失曲线图：outputs/<时间戳>/loss_curve.png
- 训练参数：outputs/<时间戳>/train_config.json
- 缩放器：shandong_scaler.pkl

## 4. 生成命令（推理）

按指定标签生成样本，并自动逆归一化为 MW：

python generate.py \
	--checkpoint outputs/<时间戳>/cgan_wgangp_504.pth \
	--label 2 \
	--num-samples 20 \
	--scaler-path shandong_scaler.pkl \
	--output generated_label2.csv

说明：

- --label 可选 0 / 1 / 2
- 输出 CSV 包含 504 列物理量（Load/Wind/Solar）和 output_label 列

## 5. 可视化命令

在一张图中显示某一条生成样本的负荷、风电、光伏曲线：

python plot_results.py \
	--input generated_label2.csv \
	--sample-index 0 \
	--output generated_curves.png

## 6. 一键运行脚本

项目提供 run_all.sh，可一次完成：训练 -> 生成 -> 绘图 -> 场景质量评估。

使用方式：

chmod +x run_all.sh

./run_all.sh \
	--data shandong_gan_ready_final.csv \
	--epochs 800 \
	--label 1 \
	--num-samples 20 \
	--gen-output generated_label1.csv \
	--plot-output generated_label1.png

脚本会自动使用最新一次训练目录中的 cgan_wgangp_504.pth 进行生成，并输出评估结果到 evaluation 目录。

常用开关说明：

- --data：训练/评估使用的真实数据 CSV（默认 shandong_gan_ready_final.csv）
- --epochs：训练轮数
- --batch-size：训练 batch 大小
- --noise-dim：生成器噪声维度
- --n-critic：每次 G 更新前的 D 更新次数
- --gp-lambda：梯度惩罚系数
- --lr：学习率
- --label：生成条件标签（0/1/2）
- --num-samples：生成样本数
- --gen-output：生成 CSV 输出路径
- --plot-output：三曲线图输出路径
- --sample-index：绘图时选取第几条生成样本
- --skip-train：跳过训练，只做生成+绘图+评估
- --checkpoint：手动指定权重文件路径（.pth）
- --no-eval：关闭评估步骤
- --eval-out-dir：评估输出目录

示例（跳过训练，直接用已有模型）：

./run_all.sh \
	--skip-train \
	--checkpoint outputs/<时间戳>/cgan_wgangp_504.pth \
	--label 2 \
	--num-samples 50

## 7. 场景生成指标与可视化

项目新增 evaluate.py，专门评估场景生成质量（不是预测误差）。

指标（metrics.json / metrics.csv）：

- load/wind/solar_w1_quantile：边缘分布一阶 Wasserstein 近似距离（越小越好）
- load/wind/solar_p1_p99_coverage：生成值落在真实分布 [P1, P99] 区间的比例
- *_mean_profile_mae：168 小时均值曲线误差（越小越好）
- *_std_profile_mae：168 小时标准差曲线误差（越小越好）
- *_ramp_mean_abs_diff、*_ramp_std_abs_diff：爬坡统计差异（越小越好）
- energy_mean_l1、energy_std_l1：周能量聚合统计差异（越小越好）
- cross_var_corr_frobenius：负荷-风-光相关矩阵差异（越小越好）
- diversity_gen_pairdist_mean / diversity_real_pairdist_mean：场景多样性对比
- diversity_ratio_gen_over_real：生成多样性与真实多样性的比值（接近 1 较好）

可视化图：

- mean_std_profiles.png：真实 vs 生成 的均值±标准差时序包络
- marginal_distributions.png：负荷/风/光边缘分布直方图对比
- cross_var_correlations.png：跨变量相关矩阵对比与差值热力图

单独运行评估：

python evaluate.py \
	--real-data shandong_gan_ready_final.csv \
	--generated-data generated_label2.csv \
	--label 2 \
	--out-dir evaluation

## 8. 标签 0/1/2 批量实验（避免覆盖）

是的，如果你一直用同一个输出路径，结果会被覆盖。
推荐使用 run_label_sweep.sh 自动为每个标签创建独立文件夹和文件名。

示例：

chmod +x run_label_sweep.sh

./run_label_sweep.sh \
	--checkpoint outputs/<时间戳>/cgan_wgangp_504.pth \
	--num-samples 100 \
	--base-dir experiments \
	--tag baseline

输出结构示例：

experiments/baseline/
├── generated_label0.csv
├── generated_label1.csv
├── generated_label2.csv
├── generated_label0.png
├── generated_label1.png
├── generated_label2.png
├── eval_label0/
├── eval_label1/
├── eval_label2/
└── metrics_summary.csv

你也可以用 summarize_metrics.py 汇总已有实验目录：

python summarize_metrics.py --root experiments/baseline --output experiments/baseline/metrics_summary.csv

## 9. 如何判断“场景生成是否好”

建议按下面优先级看：

1. 分布是否匹配：先看 load/wind/solar_w1_quantile（越小越好）
2. 时序统计是否匹配：看 *_mean_profile_mae、*_std_profile_mae（越小越好）
3. 跨变量关系是否保留：看 cross_var_corr_frobenius（越小越好）
4. 多样性是否足够：看 diversity_ratio_gen_over_real（接近 1 较好）

经验参考：

- diversity_ratio_gen_over_real < 0.7：通常偏保守，可能有模式收缩
- diversity_ratio_gen_over_real > 1.3：可能过于发散
- cross_var_corr_frobenius 越接近 0 越能说明负荷-风-光协同关系学得更好

注意：标签 0 在当前数据里样本很少（仅少量周），其指标波动会明显更大，建议在报告里单独说明。

## 10. 项目目录结构

.
├── model.py                 # 生成器/判别器（含标签 Embedding）
├── train.py                 # 训练脚本（条件 WGAN-GP）
├── generate.py              # 推理脚本（含逆归一化）
├── plot_results.py          # 结果可视化脚本
├── evaluate.py              # 场景生成指标计算与可视化
├── run_all.sh               # 一键脚本：训练+生成+绘图
├── run_label_sweep.sh       # 0/1/2 标签批量实验
├── summarize_metrics.py     # 多标签/多实验指标汇总
├── requirements.txt         # 依赖库
├── shandong_gan_ready_final.csv
├── shandong_scaler.pkl      # 训练后生成
└── outputs/                 # 训练输出目录

## 11. 关键实现说明

- 维度适配：模型输入/输出均为 504 维，完整建模负荷-风-光协同关系。
- 条件注入：Generator 和 Discriminator 都通过 Embedding 注入 output_label 条件。
- 稳定训练：使用 Wasserstein 损失 + Gradient Penalty（WGAN-GP）。
- 输出范围：Generator 最后一层使用 Sigmoid，与 0-1 归一化特征范围对齐。
