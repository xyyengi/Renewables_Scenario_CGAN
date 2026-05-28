from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "experiments" / "experiments" / "cgan_20260517_123715"
ASSET_DIR = EXP_DIR / "ppt_assets"
REPORT_PATH = EXP_DIR / "CGAN场景生成实验结果整理.md"
REAL_PATH = ROOT / "germany_data_final.csv"
GENERATION_RAW = ROOT / "Actual_generation_202301010000_202603011700_Hour (1).csv"
CONSUMPTION_RAW = ROOT / "Actual_consumption_202301010000_202603011700_Hour (2).csv"

VAR_SLICES = {
    "Load": slice(0, 168),
    "Wind": slice(168, 336),
    "Solar": slice(336, 504),
}
LABEL_NAMES = {0: "低出力", 1: "中出力", 2: "高出力"}


def load_504(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    return df.iloc[:, :504].astype(float).to_numpy(), df


def by_var(x: np.ndarray) -> dict[str, np.ndarray]:
    return {name: x[:, sl] for name, sl in VAR_SLICES.items()}


def corr3(x: np.ndarray) -> np.ndarray:
    parts = by_var(x)
    stacked = np.vstack([parts[name].reshape(-1) for name in VAR_SLICES])
    return np.corrcoef(stacked)


def quantile_l1(real: np.ndarray, gen: np.ndarray) -> float:
    qs = np.linspace(0.01, 0.99, 99)
    vals = []
    for sl in VAR_SLICES.values():
        rq = np.quantile(real[:, sl].reshape(-1), qs)
        gq = np.quantile(gen[:, sl].reshape(-1), qs)
        vals.append(np.mean(np.abs(rq - gq)))
    return float(np.mean(vals))


def pairdist_mean(x: np.ndarray, cap: int = 250) -> float:
    if len(x) < 2:
        return float("nan")
    if len(x) > cap:
        rng = np.random.default_rng(42)
        x = x[rng.choice(len(x), size=cap, replace=False)]
    return float(pdist(x, metric="euclidean").mean())


def label_stats(real_all: np.ndarray, labels: np.ndarray) -> dict[int, dict[str, float]]:
    out = {}
    for label in [0, 1, 2]:
        real = real_all[labels == label]
        gen, _ = load_504(EXP_DIR / f"generated_label{label}.csv")
        mean_maes = []
        var_maes = []
        rmses = []
        for sl in VAR_SLICES.values():
            r = real[:, sl]
            g = gen[:, sl]
            mean_maes.append(np.mean(np.abs(g.mean(axis=0) - r.mean(axis=0))))
            var_maes.append(np.mean(np.abs(g.var(axis=0) - r.var(axis=0))))
            rmses.append(np.sqrt(np.mean((g.mean(axis=0) - r.mean(axis=0)) ** 2)))
        rdiv = pairdist_mean(real)
        gdiv = pairdist_mean(gen)
        out[label] = {
            "n_real": float(len(real)),
            "n_gen": float(len(gen)),
            "mean_mae": float(np.mean(mean_maes)),
            "var_mae": float(np.mean(var_maes)),
            "rmse_mean_profile": float(np.mean(rmses)),
            "quantile_l1": quantile_l1(real, gen),
            "corr_fro": float(np.linalg.norm(corr3(real) - corr3(gen), ord="fro")),
            "diversity_ratio": float(gdiv / rdiv) if rdiv else float("nan"),
        }
        metrics_path = EXP_DIR / f"eval_label{label}" / "metrics_diffusion_style.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            out[label].update(
                {
                    "total_crps": float(metrics["total_crps"]),
                    "total_cov_80": float(metrics["wind_coverage_80%"] + metrics["solar_coverage_80%"] + metrics["load_coverage_80%"]) / 3,
                    "total_acf_mae": float(metrics["total_acf_mae"]),
                }
            )
    return out


def best_fmt(rows: dict[int, dict[str, float]], key: str, label: int, mode: str = "min") -> str:
    val = rows[label][key]
    if mode == "one":
        scores = {k: abs(v[key] - 1.0) for k, v in rows.items()}
        best = min(scores, key=scores.get)
    elif mode == "cov80":
        scores = {k: abs(v[key] - 80.0) for k, v in rows.items()}
        best = min(scores, key=scores.get)
    else:
        best = min(rows, key=lambda k: rows[k][key])
    text = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
    return f"**{text}**" if label == best else text


def raw_range(path: Path) -> tuple[str, str, int]:
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    ts = pd.to_datetime(df["Start date"], format="%b %d, %Y %I:%M %p", errors="coerce").dropna()
    return ts.min().strftime("%Y-%m-%d %H:%M"), ts.max().strftime("%Y-%m-%d %H:%M"), len(ts)


def plot_scenarios(real: np.ndarray, gen: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    hours = np.arange(168)
    for ax, (name, sl) in zip(axes, VAR_SLICES.items()):
        for i in range(min(12, len(gen))):
            ax.plot(hours, gen[i, sl], color="#2f80ed", alpha=0.25, linewidth=0.9)
        ax.plot(hours, real[:, sl].mean(axis=0), color="#111111", linewidth=2.0, label="Real mean")
        ax.set_ylabel(name.split()[0])
        ax.grid(alpha=0.22)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Hour in week")
    fig.suptitle("CGAN Generated Weekly Scenarios (label 1: medium renewable output)", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_distributions(real: np.ndarray, gen: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, sl) in zip(axes, VAR_SLICES.items()):
        ax.hist(real[:, sl].reshape(-1), bins=35, density=True, alpha=0.52, label="Real", color="#111111")
        ax.hist(gen[:, sl].reshape(-1), bins=35, density=True, alpha=0.52, label="CGAN", color="#2f80ed")
        ax.set_title(name)
        ax.set_xlabel("Normalized value")
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Real vs CGAN Marginal Distributions (label 1)")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_corr(real: np.ndarray, gen: np.ndarray, out: Path) -> None:
    mats = [corr3(real), corr3(gen), corr3(gen) - corr3(real)]
    titles = ["Real correlation", "CGAN correlation", "CGAN - Real"]
    labels = ["Wind", "Solar", "Load"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, mat, title in zip(axes, mats, titles):
        vmax = 1 if title != "CGAN - Real" else max(0.1, np.abs(mat).max())
        im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_title(title)
        ax.set_xticks(range(3), labels=labels)
        ax.set_yticks(range(3), labels=labels)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_envelope(real: np.ndarray, gen: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    hours = np.arange(168)
    for ax, (name, sl) in zip(axes, VAR_SLICES.items()):
        gp10, gp90 = np.quantile(gen[:, sl], [0.10, 0.90], axis=0)
        gp25, gp75 = np.quantile(gen[:, sl], [0.25, 0.75], axis=0)
        rp10, rp90 = np.quantile(real[:, sl], [0.10, 0.90], axis=0)
        ax.fill_between(hours, gp10, gp90, color="#2f80ed", alpha=0.22, label="CGAN P10-P90")
        ax.fill_between(hours, gp25, gp75, color="#2f80ed", alpha=0.38, label="CGAN P25-P75")
        ax.plot(hours, real[:, sl].mean(axis=0), color="#111111", linewidth=1.8, label="Real mean")
        ax.plot(hours, rp10, color="#777777", linewidth=0.8, linestyle="--", label="Real P10/P90")
        ax.plot(hours, rp90, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_ylabel(name.split()[0])
        ax.grid(alpha=0.22)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Hour in week")
    fig.suptitle("CGAN Scenario Envelope (label 1)", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def make_report() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    real_all, real_df = load_504(REAL_PATH)
    labels = real_df["output_label"].astype(int).to_numpy()
    rows = label_stats(real_all, labels)

    real_mid = real_all[labels == 1]
    gen_mid, _ = load_504(EXP_DIR / "generated_label1.csv")

    paths = {
        "scenario": ASSET_DIR / "fig1_cgan_generated_scenarios_label1.png",
        "dist": ASSET_DIR / "fig2_real_vs_cgan_distribution_label1.png",
        "corr": ASSET_DIR / "fig3_real_vs_cgan_correlation_label1.png",
        "envelope": ASSET_DIR / "fig4_cgan_scenario_envelope_label1.png",
    }
    plot_scenarios(real_mid, gen_mid, paths["scenario"])
    plot_distributions(real_mid, gen_mid, paths["dist"])
    plot_corr(real_mid, gen_mid, paths["corr"])
    plot_envelope(real_mid, gen_mid, paths["envelope"])

    gen_start, gen_end, gen_hours = raw_range(GENERATION_RAW)
    con_start, con_end, con_hours = raw_range(CONSUMPTION_RAW)
    label_counts = real_df["output_label"].value_counts().sort_index().to_dict()

    table_lines = [
        "| 出力标签 | 真实周样本数 | 生成样本数 | 均值曲线MAE↓ | 方差曲线MAE↓ | 均值曲线RMSE↓ | 分布分位数L1↓ | 相关矩阵差异Fro↓ | 多样性比值(接近1) | CRPS↓ | 80%覆盖率(接近80%) | ACF MAE↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in [0, 1, 2]:
        r = rows[label]
        table_lines.append(
            "| "
            + f"{label}（{LABEL_NAMES[label]}） | {int(r['n_real'])} | {int(r['n_gen'])} | "
            + f"{best_fmt(rows, 'mean_mae', label)} | "
            + f"{best_fmt(rows, 'var_mae', label)} | "
            + f"{best_fmt(rows, 'rmse_mean_profile', label)} | "
            + f"{best_fmt(rows, 'quantile_l1', label)} | "
            + f"{best_fmt(rows, 'corr_fro', label)} | "
            + f"{best_fmt(rows, 'diversity_ratio', label, mode='one')} | "
            + f"{best_fmt(rows, 'total_crps', label)} | "
            + f"{best_fmt(rows, 'total_cov_80', label, mode='cov80')} | "
            + f"{best_fmt(rows, 'total_acf_mae', label)} |"
        )

    mid = rows[1]
    md = f"""# CGAN场景生成实验结果整理

## 1. 实验目的

本实验在预测基线基础上，引入条件生成对抗网络（CGAN/WGAN-GP）生成风电、光伏、负荷等源荷不确定性场景，实现由单一预测结果向多场景概率表达的初步扩展。CGAN 作为本阶段深度生成模型的基线方法，其结果用于后续与 Diffusion Model 的生成质量、稳定性和场景覆盖能力进行对比。

## 2. 数据与实验设置

- 数据来源：德国 SMARD 原始小时级数据；发电数据 `{GENERATION_RAW.name}`，时间范围 {gen_start} 至 {gen_end}；负荷数据 `{CONSUMPTION_RAW.name}`，时间范围 {con_start} 至 {con_end}。
- 预处理数据：`germany_data_final.csv`，共 {len(real_df)} 个 168 小时周样本；标签分布为 0/低出力 {label_counts.get(0, 0)} 周、1/中出力 {label_counts.get(1, 0)} 周、2/高出力 {label_counts.get(2, 0)} 周。
- 生成对象：Load、Wind、Solar 三变量联合周场景；其中 Wind 为陆上风电与海上风电之和，Solar 为光伏。
- 条件信息：`output_label`，由每周 Wind+Solar 平均出力按 0.33/0.66 分位数划分为低、中、高出力；当前对比重点使用 label 1（中出力），因为样本数量均衡且运行状态更具代表性。
- 场景维度：每个样本为 504 维，即 Load(168) + Wind(168) + Solar(168)，数值为 0-1 归一化尺度。
- 随机噪声维度：128。
- 模型结构：生成器输入为噪声向量与标签 Embedding 拼接，经过 512-1024-1024 全连接层、BatchNorm 与 LeakyReLU，输出 504 维 Sigmoid 场景；判别器/critic 输入为场景与标签 Embedding 拼接，经过 1024-512-256 全连接层、LeakyReLU 与 Dropout，输出 Wasserstein score。
- 训练设置：条件 WGAN-GP，epochs=800，batch size=16，learning rate=1e-4，Adam betas=(0.5, 0.9)，n_critic=5，gradient penalty 系数=10，seed=42。
- 评价指标：均值曲线误差、方差曲线误差、均值曲线 RMSE、分布分位数 L1、相关矩阵 Frobenius 差异、多样性比值、CRPS、覆盖率、ACF MAE。

## 3. 实验结果表格

{chr(10).join(table_lines)}

说明：除“多样性比值”和“80%覆盖率”外，表中指标越小越好；多样性比值越接近 1 越好，80%覆盖率越接近 80% 越好。中出力 label 1 的分布误差为 {mid['quantile_l1']:.4f}，多样性比值为 {mid['diversity_ratio']:.4f}，说明 CGAN 已能生成一定随机波动，但生成场景整体仍偏保守，80%区间覆盖率低于理想水平。

## 4. 可视化图及图意说明

| 图号 | 图片文件路径 | 图题 | 图意说明 |
|---|---|---|---|
| 图1 | `{paths['scenario'].relative_to(ROOT)}` | CGAN生成场景曲线图（中出力） | 多条 CGAN 生成周场景与真实均值曲线对比，用于展示趋势跟随能力和随机波动范围。 |
| 图2 | `{paths['dist'].relative_to(ROOT)}` | 真实数据 vs CGAN生成数据分布对比图（中出力） | 分变量比较 Load、Wind、Solar 的边际分布，观察生成样本是否贴近真实样本分布。 |
| 图3 | `{paths['corr'].relative_to(ROOT)}` | 真实相关性 vs CGAN生成相关性热力图（中出力） | 对比真实与生成样本的 Wind、Solar、Load 跨变量相关结构，并给出差异矩阵。 |
| 图4 | `{paths['envelope'].relative_to(ROOT)}` | CGAN场景包络图（中出力） | 展示 CGAN 生成场景 P10-P90 与 P25-P75 分位数带，并叠加真实均值和真实 P10/P90 区间。 |

## 5. 可直接用于PPT的结果总结

- CGAN 能够生成具有一定随机波动特征的风电、光伏、负荷联合场景，初步实现了从单一预测向多场景表达的扩展。
- 在中出力代表性场景下，生成结果能够跟随源荷周内趋势，边际分布与真实样本具有一定接近性。
- CGAN 生成场景的覆盖率低于理想水平，场景包络偏窄，说明极端波动和场景多样性刻画仍不足。
- CGAN/WGAN-GP 对样本分布和训练参数较敏感，可作为后续 Diffusion Model 的对比基线，用于突出扩散模型在稳定性和覆盖能力上的改进。

## 6. 当前不足与后续衔接

- 当前 CGAN 仅使用出力等级标签作为条件，尚未显式引入预测值、天气特征、日历特征或历史序列，因此条件约束能力有限。
- 数据按周切片后样本量较小，各标签仅约 50 余个真实周样本，GAN 训练容易出现覆盖不足或模式收缩。
- 评价结果显示生成场景可表达一般波动，但对极端出力、尖峰负荷和跨变量相关性的还原仍需加强。
- 后续扩散模型将作为主要改进方向：通过逐步去噪生成机制提升训练稳定性，并增强多场景覆盖范围和复杂分布拟合能力。
"""
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"Report: {REPORT_PATH}")
    for p in paths.values():
        print(f"Figure: {p}")


if __name__ == "__main__":
    make_report()
