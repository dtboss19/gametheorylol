"""
Test script for Subgame Perfect Nash Equilibrium (SPNE) solver.
Demonstrates how to find optimal strategies using backward induction.
"""

from nash_equilibrium import SubgamePerfectNashEquilibrium
from composition_scorer import CompositionScorer
from typing import List, Dict, Any
import os
import random
import math
from datetime import datetime

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import draft state and champion pool from get_worlds_data
from get_worlds_data import (
    CHAMPION_POOL, ALL_CHAMPIONS, TEAM_1, TEAM_2,
    DraftState, create_example_draft
)


def select_champions_by_role(scorer: CompositionScorer, all_champions: List[str], 
                              champions_per_role: int = 5) -> List[str]:
    """
    Randomly select champions from each role.
    
    Args:
        scorer: CompositionScorer instance to check champion roles
        all_champions: List of all available champions
        champions_per_role: Number of champions to select per role (default 5)
    
    Returns:
        List of selected champions (5 per role = 25 total)
    """
    role_mapping = {
        'top': 'is_top',
        'jungle': 'is_jungle',
        'mid': 'is_mid',
        'adc': 'is_adc',
        'support': 'is_support'
    }
    
    selected_champions = []
    role_names = ['top', 'jungle', 'mid', 'adc', 'support']
    
    for role_name in role_names:
        role_key = role_mapping[role_name]
        # Get all champions that can play this role
        role_champions = []
        for champ in all_champions:
            champ_data = scorer.get_champion_data(champ)
            if champ_data:
                role_value = champ_data.get(role_key, 0)
                # Check if champion can play this role
                if role_value == 1 or role_value == '1' or str(role_value).strip() == '1':
                    role_champions.append(champ)
        
        # If no role data, check if champion has any role data at all
        # If no role data exists, include all champions as fallback
        if not role_champions:
            # Check if any champion has role data
            has_any_role_data = False
            for champ in all_chchamps:
                champ_data = scorer.get_champion_data(champ)
                if champ_data:
                    if any(champ_data.get(f'is_{r}', 0) == 1 for r in role_names):
                        has_any_role_data = True
                        break
            
            if not has_any_role_data:
                # No role data at all - use all champions
                role_champions = all_champions.copy()
        
        # Randomly select champions_per_role from this role
        if len(role_champions) >= champions_per_role:
            selected = random.sample(role_champions, champions_per_role)
        else:
            # Not enough champions for this role, take all available
            selected = role_champions.copy()
            # Fill remaining with random champions not already selected
            remaining = [c for c in all_champions if c not in selected_champions and c not in selected]
            needed = champions_per_role - len(selected)
            if remaining and needed > 0:
                selected.extend(random.sample(remaining, min(needed, len(remaining))))
        
        selected_champions.extend(selected)
        print(f"  {role_name.capitalize()}: {len(selected)} champions selected")
    
    # Remove duplicates while preserving order
    # (Some champions can play multiple roles, so they may be selected multiple times)
    seen = set()
    unique_champions = []
    for champ in selected_champions:
        if champ not in seen:
            seen.add(champ)
            unique_champions.append(champ)
    
    total_selected = len(selected_champions)
    unique_count = len(unique_champions)
    if total_selected > unique_count:
        print(f"  (Removed {total_selected - unique_count} duplicate selections - champions that play multiple roles)")
    
    return unique_champions


def test_spne_sequential_draft():
    """Test SPNE for sequential draft game."""
    print("="*70)
    print("SUBGAME PERFECT NASH EQUILIBRIUM - SEQUENTIAL DRAFT")
    print("="*70)
    
    scorer = CompositionScorer()
    # Use skip_api_calls=True for much faster runtime (uses composition scores only)
    # Set to False if you need accurate lane matchup and player comfort scores
    spne = SubgamePerfectNashEquilibrium(scorer, w1=0.35, w2=0.35, w3=0.3,
                                         skip_api_calls=True,  # Set to True for speed
                                         show_progress=True)
    
    # Get available champions (limit for performance)
    all_champions = spne.get_all_champions()
    print(f"\nTotal champions available: {len(all_champions)}")
    
    # Select 5 random champions from each role (25 total)
    print(f"\nSelecting 10 random champions per role:")
    available_champions = select_champions_by_role(scorer, all_champions, champions_per_role=30)
    print(f"\nUsing subset of {len(available_champions)} champions for computation")
    
    # Standard League of Legends draft order:
    # Blue picks 1, Red picks 2, Blue picks 2, Red picks 2, Blue picks 2, Red picks 1
    draft_order = ['blue', 'red', 'red', 'blue', 'blue', 
                   'red', 'red', 'blue', 'blue', 'red']
    
    print(f"\nDraft order: {draft_order}")
    print("Starting backward induction to find SPNE...")
    print("(This may take a while depending on number of champions)\n")
    
    try:
        # Set max_depth here to control runtime vs optimality trade-off:
        # - max_depth=3-4: Very fast (~5-30s), approximate
        # - max_depth=5-7: Moderate (~30s-2min), good balance
        # - max_depth=10: Full draft, slower but optimal
        result = spne.find_spne_sequential_draft(
            available_champions=available_champions,
            draft_order=draft_order,
            max_depth=5,  # Adjust this value to control runtime
            auto_ban_counters=False,  # Auto-ban champions with high winrate vs picked champions
            bans_per_pick=1  # Number of bans per pick (default 1)
        )
        
        print("\n" + "="*70)
        print("SPNE RESULT")
        print("="*70)
        print(f"\nBlue Team (SPNE strategy): {result['blue_team']}")
        print(f"Red Team (SPNE strategy): {result['red_team']}")
        print(f"\nPayoff (Blue advantage): {result['payoff']:+.4f}")
        print(f"  - Positive = Blue team favored")
        print(f"  - Negative = Red team favored")
        print(f"  - Zero = Balanced")
        
        if result.get('optimal_action'):
            print(f"\nOptimal first action: {result['optimal_action']}")
        
        # Show banned champions if auto-banning was enabled
        if result.get('auto_bans_enabled'):
            banned = result.get('banned_champions', [])
            if banned:
                print(f"\nAuto-banned champions ({len(banned)}): {banned}")
                print(f"  (Banned based on counter matchups vs picked champions)")
        
        # Show cache statistics if available
        if 'cache_stats' in result:
            stats = result['cache_stats']
            print(f"\nPerformance Stats:")
            print(f"  Total calculations: {stats.get('total_calculations', 0):,}")
            print(f"  Cache size: {stats.get('payoff_cache_size', 0):,}")
            print(f"  Cache hit rate: {stats.get('cache_hit_rate', 'N/A')}")
        
        # Calculate win probabilities
        S = result['payoff']
        alpha = scorer.ALPHA
        U_blue = 1 / (1 + math.exp(-(S + alpha)))
        U_red = 1 - U_blue
        
        print(f"\nWin Probabilities:")
        print(f"  Blue Team: {U_blue:.2%}")
        print(f"  Red Team:  {U_red:.2%}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


# NOTE: test_spne_partial_draft, test_spne_simultaneous, test_payoff_calculation,
# test_worlds_draft_lastpick, and related helpers have been moved to extras
# for legacy and deeper debugging workflows. They still import the same core
# modules and can be run from the project root via:
#
#   python extras/test_nash_equilibrium.py sequential
#
# (Full content preserved from the original root-level script.)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == 'sequential':
            test_spne_sequential_draft()
        else:
            print(f"Unknown test type: {test_type}")
            print("Available: sequential")
    else:
        # Default to sequential test for quick demo
        test_spne_sequential_draft()


