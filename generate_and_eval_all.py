#!/usr/bin/env python3
"""一键生成所有标签的场景并评估"""

import subprocess
import sys
import os
from datetime import datetime

def run_cmd(cmd, desc):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"🚀 {desc}")
    print(f"{'='*60}")
    print(f"命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ 失败: {desc}")
        return False
    print(f"✅ 完成: {desc}")
    return True

def main():
    # 配置
    CHECKPOINT = "outputs/20260517_122758/cgan_wgangp_504.pth"
    REAL_DATA = "germany_data_final.csv"
    NUM_SAMPLES = 1000
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiments/cgan_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# CGAN 批量生成与评估")
    print(f"# 模型: {CHECKPOINT}")
    print(f"# 每个标签生成: {NUM_SAMPLES} 个样本")
    print(f"# 输出目录: {output_dir}")
    print(f"{'#'*60}")
    
    # 生成3个标签的场景
    generated_files = []
    for label in [0, 1, 2]:
        output_file = f"{output_dir}/generated_label{label}.csv"
        cmd = f"python generate.py --checkpoint {CHECKPOINT} --label {label} --num-samples {NUM_SAMPLES} --output {output_file}"
        if not run_cmd(cmd, f"生成标签 {label} 的 {NUM_SAMPLES} 个场景"):
            sys.exit(1)
        generated_files.append((label, output_file))
    
    # 评估每个标签
    print(f"\n{'#'*60}")
    print(f"# 开始评估")
    print(f"{'#'*60}")
    
    for label, gen_file in generated_files:
        eval_dir = f"{output_dir}/eval_label{label}"
        cmd = f"python evaluate_diffusion_style.py --real-data {REAL_DATA} --generated-data {gen_file} --label {label} --out-dir {eval_dir}"
        if not run_cmd(cmd, f"评估标签 {label}"):
            print(f"⚠️ 标签 {label} 评估失败，继续下一个...")
    
    # 汇总结果
    print(f"\n{'#'*60}")
    print(f"# 汇总评估结果")
    print(f"{'#'*60}")
    
    summary_file = f"{output_dir}/metrics_summary.csv"
    with open(summary_file, 'w') as f:
        f.write("label,metric,value\n")
        
        for label in [0, 1, 2]:
            # 尝试新的文件名格式
            metrics_file = f"{output_dir}/eval_label{label}/metrics_diffusion_style.json"
            if not os.path.exists(metrics_file):
                # 回退到旧文件名
                metrics_file = f"{output_dir}/eval_label{label}/metrics.json"
            
            if os.path.exists(metrics_file):
                import json
                with open(metrics_file) as mf:
                    metrics = json.load(mf)
                    for metric_name, value in metrics.items():
                        f.write(f"{label},{metric_name},{value}\n")
                print(f"✅ 标签 {label}: 已汇总 {len(metrics)} 个指标")
            else:
                print(f"⚠️ 标签 {label}: 未找到指标文件 {metrics_file}")
    
    print(f"✅ 汇总完成: {summary_file}")
    
    # 打印最终结果
    print(f"\n{'='*60}")
    print(f"🎉 全部完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"\n生成文件:")
    for label, gen_file in generated_files:
        print(f"  - 标签{label}: {gen_file}")
    print(f"\n评估结果:")
    for label in [0, 1, 2]:
        print(f"  - 标签{label}: {output_dir}/eval_label{label}/")
    print(f"\n汇总文件: {summary_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
