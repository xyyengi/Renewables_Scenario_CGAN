# CGAN 重新训练与评估指南（德国数据 + 扩散指标）

## 📁 清理状态

已完成深度清理：
- ✅ 旧实验输出已归档到 `archive/`
- ✅ 保留核心代码文件
- ✅ 保留德国数据集 `germany_data_final.csv`
- ✅ 保留扩散模型评估结果 `cdm_eval_cpu_25w_100s/`（用于对比）

## 🚀 快速开始（3步）

### 第1步：训练模型

```bash
python train.py \
  --data germany_data_final.csv \
  --epochs 800 \
  --batch-size 16 \
  --noise-dim 128 \
  --n-critic 5 \
  --gp-lambda 10.0 \
  --lr 1e-4 \
  --save-dir outputs \
  --scaler-path germany_scaler.pkl
```

**输出位置**：`outputs/<时间戳>/cgan_wgangp_504.pth`

---

### 第2步：批量生成场景（标签0/1/2）

使用现有脚本，为每个标签生成100个场景：

```bash
# 获取最新训练好的模型
CHECKPOINT=$(ls -td outputs/*/cgan_wgangp_504.pth 2>/dev/null | head -n 1)

# 创建实验目录
mkdir -p experiments/germany_new

# 标签 0
python generate.py \
  --checkpoint "$CHECKPOINT" \
  --label 0 \
  --num-samples 100 \
  --scaler-path germany_scaler.pkl \
  --output experiments/germany_new/generated_label0.csv

# 标签 1
python generate.py \
  --checkpoint "$CHECKPOINT" \
  --label 1 \
  --num-samples 100 \
  --scaler-path germany_scaler.pkl \
  --output experiments/germany_new/generated_label1.csv

# 标签 2
python generate.py \
  --checkpoint "$CHECKPOINT" \
  --label 2 \
  --num-samples 100 \
  --scaler-path germany_scaler.pkl \
  --output experiments/germany_new/generated_label2.csv
```

---

### 第3步：扩散风格评估（与扩散模型对比）

对每个标签运行扩散风格评估：

```bash
# 标签 0
python evaluate_diffusion_style.py \
  --real-data germany_data_final.csv \
  --generated-data experiments/germany_new/generated_label0.csv \
  --label 0 \
  --n-samples 10 \
  --out-dir experiments/germany_new/eval_label0_diffusion

# 标签 1
python evaluate_diffusion_style.py \
  --real-data germany_data_final.csv \
  --generated-data experiments/germany_new/generated_label1.csv \
  --label 1 \
  --n-samples 10 \
  --out-dir experiments/germany_new/eval_label1_diffusion

# 标签 2
python evaluate_diffusion_style.py \
  --real-data germany_data_final.csv \
  --generated-data experiments/germany_new/generated_label2.csv \
  --label 2 \
  --n-samples 10 \
  --out-dir experiments/germany_new/eval_label2_diffusion
```

---

## 📊 查看结果

### 扩散模型评估结果（对比基准）
```bash
cat cdm_eval_cpu_25w_100s/metrics.json
```

### CGAN评估结果
```bash
# 标签0
cat experiments/germany_new/eval_label0_diffusion/metrics_diffusion_style.json

# 标签1
cat experiments/germany_new/eval_label1_diffusion/metrics_diffusion_style.json

# 标签2
cat experiments/germany_new/eval_label2_diffusion/metrics_diffusion_style.json
```

---

## 🔑 关键指标说明

| 指标 | 说明 | 好坏判断 |
|------|------|----------|
| `total_crps` | 连续分级概率评分 | 越小越好 |
| `total_energy_score` | 能量分数 | 越小越好 |
| `total_coverage_100%` | 100%覆盖率 | 接近100%越好 |
| `total_width_100%` | 100%区间宽度 | 适中最好 |
| `total_acf_mae` | 自相关函数MAE | 越小越好 |
| `multivariate_es` | 多变量能量分数 | 越小越好 |

---

## 📝 一键脚本（可选）

如果你想一次性完成所有步骤，可以运行：

```bash
# 1. 训练
python train.py --data germany_data_final.csv --epochs 800 --save-dir outputs --scaler-path germany_scaler.pkl

# 2. 获取检查点路径
CHECKPOINT=$(ls -td outputs/*/cgan_wgangp_504.pth 2>/dev/null | head -n 1)

# 3. 生成+评估所有标签
for label in 0 1 2; do
  echo "=== Label $label ==="
  
  # 生成
  python generate.py \
    --checkpoint "$CHECKPOINT" \
    --label $label \
    --num-samples 100 \
    --scaler-path germany_scaler.pkl \
    --output experiments/germany_new/generated_label${label}.csv
  
  # 扩散风格评估
  python evaluate_diffusion_style.py \
    --real-data germany_data_final.csv \
    --generated-data experiments/germany_new/generated_label${label}.csv \
    --label $label \
    --n-samples 10 \
    --out-dir experiments/germany_new/eval_label${label}_diffusion
done

echo "✅ 全部完成！"
```

---

## 📂 当前项目结构

```
Renewables_Scenario_CGAN/
├── 核心代码（保留）
│   ├── model.py              # Generator + Discriminator
│   ├── train.py              # 训练脚本
│   ├── generate.py           # 生成脚本
│   ├── evaluate.py           # 原评估指标
│   ├── evaluate_diffusion_style.py  # 扩散风格评估
│   ├── plot_results.py       # 可视化
│   ├── summarize_metrics.py  # 指标汇总
│   ├── run_all.sh            # 一键运行
│   └── run_label_sweep.sh    # 批量标签实验
│
├── 数据集
│   ├── germany_data_final.csv      # 德国数据（使用这个）
│   ├── germany_scaler.pkl          # 德国数据缩放器
│   ├── shandong_gan_ready_final.csv # 山东数据（备用）
│   └── shandong_scaler.pkl         # 山东数据缩放器
│
├── 对比基准（保留）
│   └── cdm_eval_cpu_25w_100s/      # 扩散模型评估结果
│
├── 旧实验归档
│   └── archive/                    # 所有旧实验输出
│
└── RESTART_GUIDE.md        # 本指南
```

---

## ⚠️ 注意事项

1. **训练时间**：800 epochs 大约需要 10-20 分钟（取决于GPU）
2. **显存需求**：约 2-4GB GPU 显存
3. **随机种子**：默认使用 seed=42，保证可复现
4. **标签分布**：德国数据中标签0样本较少，评估时波动可能较大

---

## 🎯 下一步

运行上面的命令开始训练，完成后对比 `cdm_eval_cpu_25w_100s/metrics.json` 和新生成的 `experiments/germany_new/eval_label*_diffusion/metrics_diffusion_style.json` 即可！
