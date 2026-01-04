"""
消融实验脚本：测试不同pvalue阈值下的标签修正性能

使用方法：
    python ablation_study.py --cid 2 --dataset_type ucf_ --pvalue_thresholds 0.01 0.05 0.1 0.15 0.2

或者使用默认的pvalue阈值范围：
    python ablation_study.py --cid 2 --dataset_type ucf_
"""

import argparse
import sys
import options
from generate_new_labels import run_ablation_study

def main():
    # First parse ablation-specific arguments (these are not in options.parser)
    ablation_parser = argparse.ArgumentParser(description='运行消融实验：测试不同pvalue阈值', add_help=False)
    ablation_parser.add_argument('--pvalue_thresholds', type=float, nargs='+', default=None,
                       help='要测试的pvalue阈值列表，如: --pvalue_thresholds 0.01 0.05 0.1')
    ablation_parser.add_argument('--no_score_decrease', action='store_true',
                       help='不使用分数下降条件，仅基于pvalue阈值进行修正（用于测试pvalue阈值的影响）')
    
    # Get ablation-specific args from sys.argv
    ablation_args, remaining_args = ablation_parser.parse_known_args()
    
    # Then parse remaining args using options parser
    args = options.parser.parse_args(remaining_args)
    
    # Determine dataset name based on cid
    dataset_name = f"{args.dataset_type}{args.cid}"
    
    # Run ablation study
    if ablation_args.pvalue_thresholds:
        pvalue_thresholds = ablation_args.pvalue_thresholds
        print(f"使用用户指定的pvalue阈值: {pvalue_thresholds}")
    else:
        pvalue_thresholds = None
        print("使用默认pvalue阈值范围: [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]")
    
    use_score_decrease = not ablation_args.no_score_decrease
    results = run_ablation_study(args, dataset_name, pvalue_thresholds, use_score_decrease=use_score_decrease)
    
    if results is None:
        print("消融实验失败，请检查错误信息")
        sys.exit(1)
    
    print("\n消融实验完成！")
    sys.exit(0)

if __name__ == '__main__':
    main()

# python ablation_study.py --cid 2 --dataset_type ucf_
# python ablation_study.py --cid 2 --dataset_type ucf_ --no_score_decrease