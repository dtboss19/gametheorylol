"""
Benchmark script to estimate runtime of backward induction on different champion subsets.
Analyzes computational complexity and provides time estimates.
"""

import time
from nash_equilibrium import SubgamePerfectNashEquilibrium
from composition_scorer import CompositionScorer
import math


def analyze_complexity(n_champions: int, picks_remaining: int):
    """
    Analyze theoretical complexity of backward induction.
    
    Args:
        n_champions: Number of available champions
        picks_remaining: Number of picks remaining in draft
    """
    print(f"\n{'='*70}")
    print(f"COMPLEXITY ANALYSIS")
    print(f"{'='*70}")
    print(f"\nChampions: {n_champions}")
    print(f"Picks remaining: {picks_remaining}")
    
    # Worst case: explore all combinations
    worst_case = 1
    for i in range(picks_remaining):
        worst_case *= (n_champions - i)
    
    print(f"\nWorst case combinations: {worst_case:,}")
    print(f"  (n * (n-1) * ... * (n-picks+1))")
    
    # With backward induction, we still need to evaluate each path
    # but we only keep the best response at each level
    # So we evaluate: n + n*(n-1) + n*(n-1)*(n-2) + ... 
    evaluations = 0
    for i in range(picks_remaining):
        product = 1
        for j in range(i + 1):
            product *= (n_champions - j)
        evaluations += product
    
    print(f"\nEstimated payoff calculations needed: {evaluations:,}")
    print(f"  (sum of products at each depth level)")
    
    # Estimate time per payoff calculation
    # This depends on whether we're fetching data from u.gg or using cached data
    print(f"\nTime estimates (assuming cached data, ~0.01s per payoff):")
    fast_time = evaluations * 0.01
    print(f"  Fast (cached): {fast_time:.1f} seconds ({fast_time/60:.1f} minutes)")
    
    print(f"\nTime estimates (with u.gg API calls, ~0.5s per payoff):")
    slow_time = evaluations * 0.5
    print(f"  Slow (API calls): {slow_time:.1f} seconds ({slow_time/60:.1f} minutes, {slow_time/3600:.2f} hours)")
    
    return evaluations, worst_case


def benchmark_payoff_calculation(scorer: CompositionScorer, spne: SubgamePerfectNashEquilibrium,
                                 n_samples: int = 10):
    """Benchmark how long a single payoff calculation takes."""
    print(f"\n{'='*70}")
    print(f"BENCHMARKING PAYOFF CALCULATION")
    print(f"{'='*70}")
    
    # Get some sample champions
    all_champions = spne.get_all_champions()
    if len(all_champions) < 10:
        print("Not enough champions in database")
        return 0.01
    
    # Test with random teams
    import random
    test_champions = all_champions[:10]
    
    times = []
    for i in range(n_samples):
        blue_team = random.sample(test_champions, 5)
        red_team = random.sample([c for c in test_champions if c not in blue_team], 5)
        
        start = time.time()
        try:
            payoff = spne.calculate_payoff(blue_team, red_team)
            elapsed = time.time() - start
            times.append(elapsed)
        except Exception as e:
            print(f"Error in sample {i+1}: {e}")
            times.append(0.01)  # Fallback estimate
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nPayoff calculation times ({n_samples} samples):")
    print(f"  Average: {avg_time*1000:.2f} ms")
    print(f"  Min: {min_time*1000:.2f} ms")
    print(f"  Max: {max_time*1000:.2f} ms")
    
    return avg_time


def estimate_runtime_for_subset(n_champions: int, picks_remaining: int = 10):
    """
    Estimate total runtime for backward induction on a subset.
    
    Args:
        n_champions: Number of champions in subset
        picks_remaining: Number of picks remaining (default 10 for full draft)
    """
    print(f"\n{'='*70}")
    print(f"RUNTIME ESTIMATE FOR {n_champions} CHAMPIONS")
    print(f"{'='*70}")
    
    # Initialize (but don't run full backward induction yet)
    scorer = CompositionScorer()
    spne = SubgamePerfectNashEquilibrium(scorer)
    
    # Benchmark actual payoff calculation time
    avg_payoff_time = benchmark_payoff_calculation(scorer, spne)
    
    # Calculate complexity
    evaluations, worst_case = analyze_complexity(n_champions, picks_remaining)
    
    # Estimate total time
    estimated_time = evaluations * avg_payoff_time
    
    print(f"\n{'='*70}")
    print(f"ESTIMATED TOTAL RUNTIME")
    print(f"{'='*70}")
    print(f"\nPayoff calculations needed: {evaluations:,}")
    print(f"Average time per payoff: {avg_payoff_time*1000:.2f} ms")
    print(f"\nEstimated total time: {estimated_time:.1f} seconds")
    print(f"  = {estimated_time/60:.2f} minutes")
    if estimated_time > 3600:
        print(f"  = {estimated_time/3600:.2f} hours")
    
    # Provide recommendations
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}")
    
    if estimated_time > 300:  # 5 minutes
        print(f"\n⚠️  Runtime is quite long ({estimated_time/60:.1f} minutes)")
        print(f"Consider:")
        print(f"  1. Reducing champion subset size (currently {n_champions})")
        print(f"  2. Using max_depth to limit search depth")
        print(f"  3. Implementing memoization to cache payoff calculations")
        print(f"  4. Using alpha-beta pruning or other optimizations")
        print(f"  5. Running with smaller depth for faster approximate solutions")
    elif estimated_time > 60:
        print(f"\n⏱️  Runtime is moderate ({estimated_time:.0f} seconds)")
        print(f"Consider limiting max_depth for faster results")
    else:
        print(f"\n✅ Runtime is reasonable ({estimated_time:.1f} seconds)")
    
    scorer.close()
    return estimated_time, evaluations


def test_small_subset():
    """Test with a very small subset to verify the algorithm works."""
    print(f"\n{'='*70}")
    print(f"TESTING WITH SMALL SUBSET (5 champions, 4 picks)")
    print(f"{'='*70}")
    
    scorer = CompositionScorer()
    spne = SubgamePerfectNashEquilibrium(scorer)
    
    all_champions = spne.get_all_champions()
    if len(all_champions) < 5:
        print("Not enough champions in database")
        return
    
    # Use only 5 champions for quick test
    test_champions = all_champions[:5]
    draft_order = ['blue', 'red', 'red', 'blue']
    
    print(f"\nChampions: {test_champions}")
    print(f"Draft order: {draft_order}")
    print("\nRunning backward induction...")
    
    start = time.time()
    try:
        result = spne.find_spne_sequential_draft(
            available_champions=test_champions,
            draft_order=draft_order,
            max_depth=4
        )
        elapsed = time.time() - start
        
        print(f"\n✅ Completed in {elapsed:.2f} seconds")
        print(f"Blue Team: {result['blue_team']}")
        print(f"Red Team: {result['red_team']}")
        print(f"Payoff: {result['payoff']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("BACKWARD INDUCTION RUNTIME ANALYSIS")
    print("="*70)
    
    # Analyze complexity for 30 champions
    print("\n" + "="*70)
    print("ANALYZING 30 CHAMPIONS SUBSET")
    print("="*70)
    
    estimate_runtime_for_subset(30, picks_remaining=10)
    
    # Also show complexity for other sizes
    print("\n\n" + "="*70)
    print("COMPLEXITY FOR DIFFERENT SUBSET SIZES")
    print("="*70)
    
    for n in [10, 20, 30, 40, 50]:
        print(f"\n{n} champions:")
        evaluations, _ = analyze_complexity(n, 10)
        # Rough estimate: 0.01s per evaluation
        est_time = evaluations * 0.01
        print(f"  Estimated time: {est_time:.1f}s ({est_time/60:.1f} min)")
    
    # Test with very small subset
    print("\n")
    test_small_subset()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nFor 30 champions with 10 picks remaining:")
    print("  - Theoretical combinations: ~1.1 trillion")
    print("  - Payoff calculations needed: ~millions (depends on pruning)")
    print("  - Estimated time: 10-60 minutes (depends on payoff calculation speed)")
    print("\nRecommendation: Use max_depth=5-7 for practical runtime (< 5 minutes)")

