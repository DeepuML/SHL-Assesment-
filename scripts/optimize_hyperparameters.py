""" Hyperparameter Optimization : Optimize retrieval parameters to maximize Recall@10.
Key Hyperparameters:
1. Retrieval candidates (k_candidates)
2. Test-type balancing ratios
3. Query expansion strength
4. Re-ranking strategies
"""

import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import Evaluator
from src.retrieval_engine_gemini_enhanced import EnhancedGeminiRetrievalEngine


class HyperparameterOptimizer:
    """Optimize hyperparameters for retrieval system."""

    def __init__(self, train_file: str = "data/train/train.csv"):
        """Initialize optimizer."""
        self.train_file = train_file
        self.best_params = None
        self.best_score = 0.0
        self.results = []

    def create_param_grid(self) -> List[Dict]:
        """Create grid of hyperparameters to test.
        Returns: List of parameter combinations."""
        # Define parameter ranges
        param_grid = {
            # Number of candidates to retrieve before balancing
            "k_candidates_multiplier": [2, 3, 5, 7],  # k * multiplier
            # Test-type balancing for technical queries (K-heavy)
            "balance_k_heavy": [
                {"K": 0.6, "P": 0.3, "C": 0.1},
                {"K": 0.7, "P": 0.2, "C": 0.1},
                {"K": 0.5, "P": 0.4, "C": 0.1},
            ],
            # Test-type balancing for behavioral queries (P-heavy)
            "balance_p_heavy": [
                {"K": 0.2, "P": 0.6, "C": 0.2},
                {"K": 0.3, "P": 0.6, "C": 0.1},
                {"K": 0.1, "P": 0.7, "C": 0.2},
            ],
            # Test-type balancing for cognitive queries (C-heavy)
            "balance_c_heavy": [
                {"K": 0.2, "P": 0.2, "C": 0.6},
                {"K": 0.1, "P": 0.3, "C": 0.6},
                {"K": 0.2, "P": 0.1, "C": 0.7},
            ],
            # Balanced distribution
            "balance_mixed": [
                {"K": 0.4, "P": 0.4, "C": 0.2},
                {"K": 0.35, "P": 0.35, "C": 0.3},
                {"K": 0.33, "P": 0.34, "C": 0.33},
            ],
            # Intent detection threshold
            "intent_threshold": [0.4, 0.5, 0.6],
        }

        # Generate all combinations
        param_combinations = []

        for k_mult in param_grid["k_candidates_multiplier"]:
            for balance_k in param_grid["balance_k_heavy"]:
                for balance_p in param_grid["balance_p_heavy"]:
                    for balance_c in param_grid["balance_c_heavy"]:
                        for balance_mix in param_grid["balance_mixed"]:
                            for threshold in param_grid["intent_threshold"]:
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

        return param_combinations

    def evaluate_params(
        self, params: Dict, engine: EnhancedGeminiRetrievalEngine
    ) -> float:
        """Evaluate single parameter configuration.
        Args:
            params: Hyperparameters to test
            engine: Retrieval engine instance
        Returns:
            Recall@10 score
        """
        # Temporarily modify engine parameters
        original_get_balanced = engine.balance_results
        original_detect_intent = engine.detect_query_intent

        # Override balance_results with new parameters
        def custom_balance(results, intent, k=10):
            threshold = params["intent_threshold"]

            # Determine which balance to use based on intent
            if intent["K"] > threshold:
                target_ratios = params["balance_k_heavy"]
            elif intent["P"] > threshold:
                target_ratios = params["balance_p_heavy"]
            elif intent["C"] > threshold:
                target_ratios = params["balance_c_heavy"]
            else:
                target_ratios = params["balance_mixed"]

            # Apply balancing
            by_type = {"K": [], "P": [], "C": [], "A": []}
            for r in results:
                test_type = r.get("test_type", "A")
                by_type[test_type].append(r)

            # Calculate target counts
            target_counts = {
                "K": int(k * target_ratios["K"]),
                "P": int(k * target_ratios["P"]),
                "C": int(k * target_ratios["C"]),
            }

            balanced = []
            for test_type, count in target_counts.items():
                balanced.extend(by_type[test_type][:count])

            # Fill remaining
            remaining = k - len(balanced)
            if remaining > 0:
                all_remaining = [r for r in results if r not in balanced]
                balanced.extend(all_remaining[:remaining])

            return balanced[:k]

        # Override recommend method
        original_recommend = engine.recommend

        def custom_recommend(query, k=10):
            query_embedding = engine.embed_query(query)

            # Use k_candidates_multiplier
            k_candidates = k * params["k_candidates_multiplier"]
            distances, indices = engine.embedding_engine.index.search(
                query_embedding,
                min(k_candidates, len(engine.embedding_engine.metadata)),
            )

            results = []
            for idx in indices[0]:
                if idx < len(engine.embedding_engine.metadata):
                    results.append(engine.embedding_engine.metadata[idx])

            intent = engine.detect_query_intent(query)
            balanced = custom_balance(results, intent, k)

            return balanced

        # Temporarily replace methods
        engine.balance_results = custom_balance
        engine.recommend = custom_recommend

        # Evaluate
        evaluator = Evaluator(engine)
        results = evaluator.evaluate(self.train_file, k=10)
        recall = results["mean_recall"]

        # Restore original methods
        engine.balance_results = original_get_balanced
        engine.recommend = original_recommend

        return recall

    def optimize(self, top_n: int = 10) -> Dict:
        """Run hyperparameter optimization.
        Args: top_n: Number of top configurations to return
        Returns:
            Best parameters and results
        """
        print("HYPERPARAMETER OPTIMIZATION")

        # Load engine once
        print("\nLoading Enhanced Gemini engine...")
        engine = EnhancedGeminiRetrievalEngine()

        # Create parameter grid
        param_combinations = self.create_param_grid()
        total_combinations = len(param_combinations)

        print(f"\nTesting {total_combinations} parameter combinations...")
        print("(This may take 10-15 minutes)\n")

        # Test each combination
        for idx, params in enumerate(param_combinations, 1):
            try:
                recall = self.evaluate_params(params, engine)

                self.results.append({"params": params, "recall": recall})

                # Update best
                if recall > self.best_score:
                    self.best_score = recall
                    self.best_params = params
                    print(f" New best! Recall@10: {recall:.4f} ({recall*100:.2f}%)")

                # Progress update every 10%
                if idx % max(1, total_combinations // 10) == 0:
                    progress = (idx / total_combinations) * 100
                    print(f"Progress: {progress:.0f}% ({idx}/{total_combinations})")
                    print(f"  Best so far: {self.best_score:.4f}")

            except Exception as e:
                print(f"  Error with params {idx}: {e}")
                continue

        # Sort results
        self.results.sort(key=lambda x: x["recall"], reverse=True)

        # Print top configurations
        print("\n" + "=" * 70)
        print(f"TOP {top_n} CONFIGURATIONS")
        print("=" * 70)

        for i, result in enumerate(self.results[:top_n], 1):
            print(
                f"\n{i}. Recall@10: {result['recall']:.4f} ({result['recall']*100:.2f}%)"
            )
            print(f"   Params:")
            params = result["params"]
            print(
                f"     - k_candidates_multiplier: {params['k_candidates_multiplier']}"
            )
            print(f"     - intent_threshold: {params['intent_threshold']}")
            print(f"     - balance_k_heavy: {params['balance_k_heavy']}")
            print(f"     - balance_p_heavy: {params['balance_p_heavy']}")

        # Save results
        output_file = "data/hyperparameter_optimization_results.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "best_params": self.best_params,
                    "best_score": self.best_score,
                    "top_10": self.results[:top_n],
                    "all_results": self.results,
                },
                f,
                indent=2,
            )

        print(f"\n\nResults saved to: {output_file}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "improvement": (self.best_score - 0.2011) / 0.2011 * 100,  # vs current best
        }


def main():
    """Run hyperparameter optimization."""
    optimizer = HyperparameterOptimizer()

    results = optimizer.optimize(top_n=10)
    print("OPTIMIZATION COMPLETE")
    print(
        f"\nBest Recall@10: {results['best_score']:.4f} ({results['best_score']*100:.2f}%)"
    )
    print(f"Current Best: 0.2011 (20.11%)")
    print(f"Improvement: {results['improvement']:+.2f}%")

    print("\nBest Parameters:")
    for key, value in results["best_params"].items():
        print(f"  {key}: {value}")

    print("\n Use these parameters in the final model for best performance!")


if __name__ == "__main__":
    main()
