"""
Quick Hyperparameter Optimization
==================================
Fast version - tests key parameters only (5-10 minutes).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.optimize_hyperparameters import HyperparameterOptimizer


class QuickOptimizer(HyperparameterOptimizer):
    """Quick version with reduced search space."""

    def create_param_grid(self):
        """Reduced parameter grid for faster optimization."""
        param_combinations = []

        # Key parameters to test
        k_mults = [3, 5]  # vs full: [2, 3, 5, 7]
        thresholds = [0.5]  # vs full: [0.4, 0.5, 0.6]

        # Best guesses based on current performance
        balances = {
            "k_heavy": [
                {"K": 0.6, "P": 0.3, "C": 0.1},
                {"K": 0.7, "P": 0.2, "C": 0.1},
            ],
            "p_heavy": [
                {"K": 0.2, "P": 0.6, "C": 0.2},
                {"K": 0.3, "P": 0.6, "C": 0.1},
            ],
            "c_heavy": [
                {"K": 0.2, "P": 0.2, "C": 0.6},
            ],
            "mixed": [
                {"K": 0.4, "P": 0.4, "C": 0.2},
                {"K": 0.33, "P": 0.34, "C": 0.33},
            ],
        }

        for k_mult in k_mults:
            for balance_k in balances["k_heavy"]:
                for balance_p in balances["p_heavy"]:
                    for balance_c in balances["c_heavy"]:
                        for balance_mix in balances["mixed"]:
                            for threshold in thresholds:
                                param_combinations.append(
                                    {
                                        "k_candidates_multiplier": k_mult,
                                        "balance_k_heavy": balance_k,
                                        "balance_p_heavy": balance_p,
                                        "balance_c_heavy": balance_c,
                                        "balance_mixed": balance_mix,
                                        "intent_threshold": threshold,
                                    }
                                )

        print(
            f"Quick mode: Testing {len(param_combinations)} combinations (vs 324 in full mode)"
        )
        return param_combinations


def main():
    """Run quick optimization."""
    print("=" * 70)
    print("QUICK HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print("\nMode: Quick (8 combinations)")
    print("Time: ~5 minutes")
    print("For full optimization (324 combinations), use optimize_hyperparameters.py\n")

    optimizer = QuickOptimizer()
    results = optimizer.optimize(top_n=5)

    print("\n" + "=" * 70)
    print("QUICK OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(
        f"\nBest Recall@10: {results['best_score']:.4f} ({results['best_score']*100:.2f}%)"
    )
    print(f"Current Best: 0.2011 (20.11%)")

    if results["improvement"] > 0:
        print(f"Improvement: +{results['improvement']:.2f}%")
    else:
        print(f"No improvement (current params already optimal)")

    print("\nRecommendation:")
    if results["improvement"] > 2:
        print(
            "  Significant improvement found! Run full optimization for even better results."
        )
    elif results["improvement"] > 0:
        print("  Marginal improvement. Current params are near-optimal.")
    else:
        print("  Current parameters are already well-tuned!")


if __name__ == "__main__":
    main()
