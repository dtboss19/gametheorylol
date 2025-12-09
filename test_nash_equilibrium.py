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
            for champ in all_champions:
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
        import math
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


def test_spne_partial_draft():
    """Test SPNE with partial teams (some picks already made)."""
    print("="*70)
    print("SUBGAME PERFECT NASH EQUILIBRIUM - PARTIAL DRAFT")
    print("="*70)
    
    scorer = CompositionScorer()
    spne = SubgamePerfectNashEquilibrium(scorer)
    
    # Get available champions
    all_champions = spne.get_all_champions()
    available_champions = all_champions[:40]
    
    # Assume some picks already made
    blue_team = ['Jax', 'Lee Sin', None, None, None]  # Top, Jungle picked
    red_team = ['Garen', None, 'Zed', None, None]  # Top, Mid picked
    
    print(f"\nInitial state:")
    print(f"Blue Team (partial): {blue_team}")
    print(f"Red Team (partial): {red_team}")
    
    # Remaining draft order
    draft_order = ['blue', 'blue', 'red', 'red', 'blue', 'red']
    
    print(f"\nRemaining draft order: {draft_order}")
    print("Finding optimal continuation...\n")
    
    try:
        result = spne.find_spne_sequential_draft(
            available_champions=available_champions,
            draft_order=draft_order,
            blue_team=blue_team,
            red_team=red_team,
            max_depth=7
        )
        
        print("\n" + "="*70)
        print("SPNE CONTINUATION")
        print("="*70)
        print(f"\nBlue Team (complete): {result['blue_team']}")
        print(f"Red Team (complete): {result['red_team']}")
        print(f"\nPayoff: {result['payoff']:+.4f}")
        
        if result.get('optimal_action'):
            print(f"\nOptimal next pick: {result['optimal_action']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


def test_spne_simultaneous():
    """Test Nash equilibrium for simultaneous game (all picks at once)."""
    print("="*70)
    print("NASH EQUILIBRIUM - SIMULTANEOUS GAME")
    print("="*70)
    
    scorer = CompositionScorer()
    spne = SubgamePerfectNashEquilibrium(scorer)
    
    # Get available champions
    all_champions = spne.get_all_champions()
    available_champions = all_champions[:30]
    
    print(f"\nUsing {len(available_champions)} champions")
    print("Finding Nash equilibrium using best response dynamics...\n")
    
    try:
        result = spne.find_spne_simultaneous(
            available_champions=available_champions
        )
        
        print("\n" + "="*70)
        print("NASH EQUILIBRIUM RESULT")
        print("="*70)
        print(f"\nBlue Team: {result['blue_team']}")
        print(f"Red Team: {result['red_team']}")
        print(f"\nPayoff: {result['payoff']:+.4f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Converged: {result['converged']}")
        
        # Calculate win probabilities
        import math
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


def test_payoff_calculation():
    """Test payoff calculation for given teams."""
    print("="*70)
    print("PAYOFF CALCULATION TEST")
    print("="*70)
    
    scorer = CompositionScorer()
    spne = SubgamePerfectNashEquilibrium(scorer)
    
    # Example teams
    blue_team = ['Jax', 'Lee Sin', 'Zed', 'Jinx', 'Thresh']
    red_team = ['Garen', 'Amumu', 'Ahri', 'Caitlyn', 'Lulu']
    
    print(f"\nBlue Team: {blue_team}")
    print(f"Red Team: {red_team}")
    print("\nCalculating payoff...\n")
    
    try:
        payoff = spne.calculate_payoff(blue_team, red_team)
        
        print("="*70)
        print("PAYOFF RESULT")
        print("="*70)
        print(f"\nPayoff (Blue advantage): {payoff:+.4f}")
        print(f"  S = w1(LaneMatchups) + w2(Comfort) + w3(TeamComp)")
        
        # Calculate win probabilities
        import math
        alpha = scorer.ALPHA
        U_blue = 1 / (1 + math.exp(-(payoff + alpha)))
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


def generate_random_draft(verbose: bool = True) -> DraftState:
    """
    Generate a randomized draft state with ROLE-BASED picks:
    - Random 10 bans (2 per role)
    - Blue has 3 locked picks, 2 open roles (picks both before Red's last)
    - Red has 4 locked picks, 1 open role (picks last, after seeing Blue's picks)
    
    This reflects the actual draft order where Blue picks twice, then Red picks last.
    """
    draft = DraftState()
    
    # Random bans: 2 from each role
    bans = []
    for role in ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]:
        role_champs = CHAMPION_POOL[role].copy()
        if len(role_champs) >= 2:
            banned = random.sample(role_champs, 2)
            bans.extend(banned)
    
    draft.set_bans(bans)
    
    if verbose:
        print(f"\n[BANS] Random Bans (2 per role):")
        print(f"   {bans}")
    
    roles = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]
    
    # Blue (Team 1) has 2 open roles to pick
    blue_open_roles = random.sample(roles, 2)
    # Red (Team 2) has 1 open role to pick (the last pick)
    red_open_role = random.choice(roles)
    
    # Get non-banned champions by role
    remaining_by_role = {}
    for role, champs in CHAMPION_POOL.items():
        remaining_by_role[role] = [c for c in champs if c not in bans]
    
    # Track all picked champions to avoid duplicates
    all_picked = []
    
    # Team 1 (Blue) picks: 3 locked roles (the ones NOT in blue_open_roles)
    team1_role_picks = {}
    for role in roles:
        if role not in blue_open_roles:
            available = [c for c in remaining_by_role[role] if c not in all_picked]
            if available:
                pick = random.choice(available)
                team1_role_picks[role] = pick
                all_picked.append(pick)
    
    # Team 2 (Red) picks: 4 locked roles (all except red_open_role)
    team2_role_picks = {}
    for role in roles:
        if role != red_open_role:
            available = [c for c in remaining_by_role[role] if c not in all_picked]
            if available:
                pick = random.choice(available)
                team2_role_picks[role] = pick
                all_picked.append(pick)
    
    # Store the open roles
    draft.team1_open_roles = blue_open_roles  # Blue has 2 open
    draft.team2_last_role = red_open_role     # Red has 1 open
    
    draft.set_team1_picks(team1_role_picks, blue_open_roles[0])  # Set first open role
    draft.set_team2_picks(team2_role_picks, red_open_role)
    
    if verbose:
        print(f"\n[DRAFT] Randomized Draft (SEQUENTIAL - Blue picks 2, Red picks 1 last):")
        print(f"\n   {TEAM_1['name']} (Blue) - 3 locked, 2 OPEN:")
        for role in roles:
            if role in blue_open_roles:
                print(f"      {role:<8}: [OPEN - picks before Red]")
            else:
                print(f"      {role:<8}: {team1_role_picks.get(role, 'N/A')}")
        print(f"   Open roles: {blue_open_roles}")
        
        print(f"\n   {TEAM_2['name']} (Red) - 4 locked, 1 OPEN:")
        for role in roles:
            if role == red_open_role:
                print(f"      {role:<8}: [OPEN - picks LAST after seeing Blue]")
            else:
                print(f"      {role:<8}: {team2_role_picks.get(role, 'N/A')}")
        print(f"   Open role: {red_open_role}")
    
    return draft


def log_composition_details(scorer: CompositionScorer, team_name: str, team: List[str], w3: float = 0.3):
    """Log detailed composition score breakdown."""
    print(f"\n   {team_name} Composition Analysis:")
    print(f"      Champions: {team}")
    
    try:
        comp = scorer.calculate_composition_score(team, w3)
        
        print(f"\n      🏆 BONUSES:")
        if 'bonuses' in comp:
            for category, value in comp['bonuses'].items():
                if value > 0:
                    print(f"         +{value:.4f} {category}")
        
        print(f"\n      ⚠️  PENALTIES:")
        if 'penalties' in comp:
            for category, value in comp['penalties'].items():
                if value > 0:
                    print(f"         -{value:.4f} {category}")
        
        print(f"\n      📈 STATS:")
        if 'stats' in comp:
            stats = comp['stats']
            print(f"         Wave Clear: {stats.get('wave_clear_count', 'N/A')}")
            print(f"         Frontline: {stats.get('frontline_count', 'N/A')}")
            print(f"         AP/AD Mix: {stats.get('ap_count', 'N/A')} AP / {stats.get('ad_count', 'N/A')} AD")
            print(f"         Poke: {stats.get('poke_count', 'N/A')}")
            print(f"         Scaling: {stats.get('scaling_count', 'N/A')}")
            print(f"         Team Fight: {stats.get('team_fight_count', 'N/A')}")
            print(f"         Hypercarry: {stats.get('hypercarry_count', 'N/A')}")
        
        print(f"\n      TOTAL COMPOSITION SCORE: {comp['total_score']:+.4f}")
        return comp
        
    except Exception as e:
        print(f"      [ERROR] Error calculating composition: {e}")
        return None


def log_matchup_details(scorer: CompositionScorer, blue_team: List[str], red_team: List[str], w1: float = 0.35):
    """Log detailed lane matchup breakdown."""
    print(f"\n   Lane Matchup Analysis:")
    print(f"      (Blue WR = Blue champion's winrate when facing Red champion, data from u.gg)")
    
    role_order = ['TOP', 'JUNGLE', 'MID', 'BOT', 'SUPPORT']
    total_diff = 0.0
    valid_matchups = 0
    
    try:
        # Try to get matchup details for each role
        for i, role in enumerate(role_order):
            if i < len(blue_team) and i < len(red_team):
                blue_champ = blue_team[i]
                red_champ = red_team[i]
                
                # u.gg counter pages show the OPPONENT's counter rate, not the champion's WR
                # e.g., Renekton's page shows "vs Rumble: 57%" meaning Rumble counters Renekton
                # So we need to INVERT: Renekton's actual WR = 1 - 57% = 43%
                scraped_rate = scorer.get_champion_matchup_winrate(blue_champ, red_champ, role.lower())
                
                if scraped_rate is not None and 0.25 <= scraped_rate <= 0.75:
                    blue_wr = 1.0 - scraped_rate  # Invert: our WR = 1 - opponent's counter rate
                else:
                    blue_wr = None
                
                if blue_wr is not None and 0.25 <= blue_wr <= 0.75:
                    # Valid matchup data
                    red_wr = 1.0 - blue_wr
                    diff = blue_wr - 0.5  # Advantage from 50%
                    total_diff += diff
                    valid_matchups += 1
                    
                    # Determine advantage
                    if blue_wr > 0.52:
                        advantage = "BLUE +"
                    elif blue_wr < 0.48:
                        advantage = "RED +"
                    else:
                        advantage = "EVEN"
                    
                    print(f"      {role:<8} {blue_champ:<12} vs {red_champ:<12} | {blue_wr:.1%} - {red_wr:.1%} | {advantage}")
                else:
                    # No valid data - assume neutral (50-50)
                    print(f"      {role:<8} {blue_champ:<12} vs {red_champ:<12} | 50.0% - 50.0% | EVEN (no data)")
        
        # Calculate weighted matchup score
        if valid_matchups > 0:
            avg_diff = total_diff / 5  # Always divide by 5 roles
            matchup_score = w1 * avg_diff
            matchup_score = max(-0.15, min(0.15, matchup_score))  # Clamp
        else:
            matchup_score = 0.0
        
        print(f"\n      TOTAL MATCHUP SCORE: {matchup_score:+.4f}")
        print(f"         (Based on {valid_matchups}/5 matchups with valid data)")
        return matchup_score
            
    except Exception as e:
        print(f"      [ERROR] Error calculating matchups: {e}")
        return 0.0


def log_player_comfort_details(blue_team: List[str], red_team: List[str],
                               team1_name: str, team2_name: str, w2: float = 0.35) -> float:
    """Log detailed player comfort breakdown using worlds data."""
    from get_worlds_data import get_player_comfort, ROLE_ORDER
    
    print(f"\n   🎮 Player Comfort Analysis:")
    
    # Get player rosters
    blue_players = list(TEAM_1['players'].values())
    red_players = list(TEAM_2['players'].values())
    
    total_comfort_diff = 0.0
    
    for i, role in enumerate(ROLE_ORDER):
        if i < len(blue_team) and i < len(red_team):
            blue_champ = blue_team[i]
            red_champ = red_team[i]
            blue_player = blue_players[i] if i < len(blue_players) else "Unknown"
            red_player = red_players[i] if i < len(red_players) else "Unknown"
            
            # Get comfort data
            blue_comfort = get_player_comfort(blue_player, blue_champ)
            red_comfort = get_player_comfort(red_player, red_champ)
            
            blue_wr = blue_comfort['winrate']
            red_wr = red_comfort['winrate']
            blue_games = blue_comfort['games']
            red_games = red_comfort['games']
            blue_found = blue_comfort['found']
            red_found = red_comfort['found']
            
            diff = blue_wr - red_wr
            total_comfort_diff += diff
            
            # Format display
            blue_wr_str = f"{blue_wr:.0%}" if blue_found else "50%*"
            red_wr_str = f"{red_wr:.0%}" if red_found else "50%*"
            blue_games_str = f"({blue_games}G)" if blue_found else "(no data)"
            red_games_str = f"({red_games}G)" if red_found else "(no data)"
            
            advantage = "BLUE" if diff > 0.02 else ("RED" if diff < -0.02 else "EVEN")
            
            print(f"      {role:<8} {blue_player[:15]:<15} {blue_champ:<12} {blue_wr_str} {blue_games_str:<10}")
            print(f"               vs {red_player[:15]:<15} {red_champ:<12} {red_wr_str} {red_games_str:<10} | {advantage}")
    
    # Calculate weighted comfort score
    avg_comfort_diff = total_comfort_diff / len(ROLE_ORDER)
    comfort_score = w2 * avg_comfort_diff
    comfort_score = max(-0.15, min(0.15, comfort_score))  # Clamp
    
    print(f"\n      TOTAL COMFORT SCORE: {comfort_score:+.4f}")
    print(f"         (Sum of differences: {total_comfort_diff:+.4f}, Avg: {avg_comfort_diff:+.4f})")
    
    return comfort_score


def log_full_payoff_breakdown(scorer: CompositionScorer, blue_team: List[str], red_team: List[str],
                              team1_name: str, team2_name: str, w1: float = 0.35, w2: float = 0.35, w3: float = 0.3,
                              pre_calculated_payoff: float = None):
    """
    Log complete payoff breakdown with all components.
    
    Note: The main analysis uses skip_api_calls=True for speed, so it only considers
    composition scores. This detailed breakdown fetches real matchup data for display
    purposes but may differ from the fast calculation.
    """
    from get_worlds_data import ROLE_ORDER
    
    print("\n" + "="*70)
    print("DETAILED PAYOFF BREAKDOWN (with full data)")
    print("="*70)
    
    if pre_calculated_payoff is not None:
        print(f"\n   ⚡ Fast calculation payoff (composition only): {pre_calculated_payoff:+.4f}")
        print(f"   Below shows full breakdown including matchups & comfort:")
    
    # Show team compositions by ROLE
    print("\n" + "-"*70)
    print("TEAM COMPOSITIONS (by role)")
    print("-"*70)
    
    print(f"\n   {team1_name} (Blue):")
    for i, role in enumerate(ROLE_ORDER):
        if i < len(blue_team):
            print(f"      {role:<8}: {blue_team[i]}")
    
    print(f"\n   {team2_name} (Red):")
    for i, role in enumerate(ROLE_ORDER):
        if i < len(red_team):
            print(f"      {role:<8}: {red_team[i]}")
    
    # Composition scores
    print("\n" + "-"*70)
    print("COMPOSITION SCORES (w3 weighted)")
    print("-"*70)
    
    blue_comp = log_composition_details(scorer, team1_name, blue_team, w3)
    red_comp = log_composition_details(scorer, team2_name, red_team, w3)
    
    comp_advantage = 0.0
    if blue_comp and red_comp:
        comp_advantage = blue_comp['total_score'] - red_comp['total_score']
        print(f"\n   Composition Advantage: {team1_name if comp_advantage > 0 else team2_name} ({comp_advantage:+.4f})")
    
    # Lane matchups
    print("\n" + "-"*70)
    print("LANE MATCHUPS (w1 weighted) - from u.gg")
    print("-"*70)
    
    matchup_score = log_matchup_details(scorer, blue_team, red_team, w1)
    
    # Player comfort
    print("\n" + "-"*70)
    print("PLAYER COMFORT (w2 weighted) - from worlds2025.db")
    print("-"*70)
    
    comfort_score = log_player_comfort_details(blue_team, red_team, team1_name, team2_name, w2)
    
    # Total payoff
    print("\n" + "-"*70)
    print("TOTAL PAYOFF CALCULATION")
    print("-"*70)
    
    total_payoff = matchup_score + comfort_score + comp_advantage
    
    print(f"\n   S = w1(Matchups) + w2(Comfort) + w3(Comp)")
    print(f"   S = {matchup_score:+.4f} + {comfort_score:+.4f} + {comp_advantage:+.4f}")
    print(f"   S = {total_payoff:+.4f}")
    
    if pre_calculated_payoff is not None:
        print(f"\n   ⚡ Note: Fast mode used Comp only: {pre_calculated_payoff:+.4f}")
        print(f"   Full calculation with all data: {total_payoff:+.4f}")
    
    # Win probability
    alpha = scorer.ALPHA
    U_blue = 1 / (1 + math.exp(-(total_payoff + alpha)))
    U_red = 1 - U_blue
    
    print(f"\n   Win Probabilities (full data):")
    print(f"   P({team1_name}) = 1 / (1 + e^-(S + α)) = {U_blue:.1%}")
    print(f"   P({team2_name}) = 1 - P({team1_name}) = {U_red:.1%}")
    
    return total_payoff


def test_worlds_draft_lastpick():
    """
    Test SPNE for Worlds 2025 draft scenario.
    Generates a RANDOM draft state and finds optimal last picks.
    Uses ROLE-BASED team building for correct lane matchups.
    """
    print("="*70)
    print("WORLDS 2025 - LAST PICK ANALYSIS (RANDOMIZED)")
    print("="*70)
    
    # Show champion pool
    print("\n[CHAMPION POOL] (30 champions):")
    for role, champs in CHAMPION_POOL.items():
        print(f"  {role}: {', '.join(champs)}")
    
    # Show teams
    print(f"\n[TEAM 1] {TEAM_1['name']}")
    for role, player in TEAM_1['players'].items():
        print(f"    {role}: {player}")
    
    print(f"\n[TEAM 2] {TEAM_2['name']}")
    for role, player in TEAM_2['players'].items():
        print(f"    {role}: {player}")
    
    # Load player comfort data
    print("\n" + "="*70)
    print("LOADING PLAYER COMFORT DATA")
    print("="*70)
    
    from get_worlds_data import load_player_comfort_data, get_player_comfort, ROLE_ORDER
    load_player_comfort_data()
    
    # Generate RANDOM draft state
    print("\n" + "="*70)
    print("GENERATING RANDOM DRAFT STATE")
    print("="*70)
    
    draft = generate_random_draft(verbose=True)
    
    # Show banned champions
    print(f"\n[BANNED] Champions ({len(draft.bans)}):")
    print(f"   {draft.bans}")
    
    # Get available champions for each team's open roles
    # Blue has 2 open roles, Red has 1
    blue_open_roles = draft.team1_open_roles
    red_open_role = draft.team2_last_role
    
    blue_available = {role: draft.get_available_champions(role) for role in blue_open_roles}
    red_available = draft.get_available_champions(red_open_role)
    
    print(f"\n   {TEAM_1['name']} (Blue) open roles:")
    for role in blue_open_roles:
        print(f"      {role}: {blue_available[role]}")
    print(f"\n   {TEAM_2['name']} (Red) open role:")
    print(f"      {red_open_role}: {red_available}")
    
    # Check if we have valid options
    if not all(blue_available.values()) or not red_available:
        print("\n[ERROR] Not enough champions available for last picks. Try again.")
        return
    
    # Initialize scorer
    scorer = CompositionScorer()
    
    print("\n" + "="*70)
    print("SEQUENTIAL GAME ANALYSIS (Backward Induction)")
    print("="*70)
    print(f"\n   Game Structure:")
    print(f"      1. Blue picks 2 champions (roles: {blue_open_roles})")
    print(f"      2. Red OBSERVES Blue's picks")
    print(f"      3. Red picks 1 champion (role: {red_open_role}) as best response")
    
    # Get player rosters as ordered lists [TOP, JG, MID, BOT, SUP]
    from get_worlds_data import ROLE_ORDER, get_player_comfort
    blue_players = [TEAM_1['players'][role] for role in ROLE_ORDER]
    red_players = [TEAM_2['players'][role] for role in ROLE_ORDER]
    
    # Weight factors
    w1, w2, w3 = 0.35, 0.35, 0.3
    
    def calculate_full_payoff(blue_team, red_team, blue_players, red_players):
        """
        Calculate payoff using ALL local data sources consistently:
        - Matchups: from u.gg (via scorer)
        - Comfort: from worlds2025.db (via get_player_comfort)
        - Composition: from lolchampiontags.db (via scorer)
        """
        # 1. Calculate composition scores
        try:
            blue_comp = scorer.calculate_composition_score(blue_team, w3)
            red_comp = scorer.calculate_composition_score(red_team, w3)
            comp_advantage = blue_comp['total_score'] - red_comp['total_score']
        except:
            comp_advantage = 0.0
        
        # 2. Calculate lane matchup score
        # u.gg shows OPPONENT's counter rate, so we invert to get our champion's WR
        matchup_score = 0.0
        valid_matchups = 0
        for i, role in enumerate(ROLE_ORDER):
            if i < len(blue_team) and i < len(red_team):
                scraped_rate = scorer.get_champion_matchup_winrate(blue_team[i], red_team[i], role.lower())
                if scraped_rate and 0.25 <= scraped_rate <= 0.75:
                    blue_wr = 1.0 - scraped_rate  # Invert: our WR = 1 - opponent's counter rate
                    matchup_score += (blue_wr - 0.5)
                    valid_matchups += 1
        if valid_matchups > 0:
            matchup_score = w1 * (matchup_score / 5)  # Average across 5 roles
            matchup_score = max(-0.15, min(0.15, matchup_score))
        
        # 3. Calculate player comfort score from worlds2025.db
        comfort_diff = 0.0
        for i, role in enumerate(ROLE_ORDER):
            if i < len(blue_team) and i < len(red_team):
                blue_comfort = get_player_comfort(blue_players[i], blue_team[i])
                red_comfort = get_player_comfort(red_players[i], red_team[i])
                comfort_diff += blue_comfort['winrate'] - red_comfort['winrate']
        comfort_score = w2 * (comfort_diff / 5)  # Average across 5 roles
        comfort_score = max(-0.15, min(0.15, comfort_score))
        
        # Total payoff
        return matchup_score + comfort_score + comp_advantage
    
    # Generate all Blue combinations (2 picks)
    from itertools import product
    role1, role2 = blue_open_roles
    blue_combinations = list(product(blue_available[role1], blue_available[role2]))
    
    print(f"\nEvaluating {len(blue_combinations)} Blue combinations x {len(red_available)} Red responses...\n")
    
    # BACKWARD INDUCTION: For each Blue combination, find Red's best response
    blue_analysis = []
    all_results = []
    
    for blue_combo in blue_combinations:
        blue_picks = {role1: blue_combo[0], role2: blue_combo[1]}
        blue_team_partial = draft.get_complete_team1_multi(blue_picks)
        
        # Find Red's best response to this Blue combination
        red_responses = []
        for red_pick in red_available:
            # Skip if Red pick conflicts with Blue picks
            if red_pick in blue_combo:
                continue
                
            red_team = draft.get_complete_team2(red_pick)
            
            # Validate teams
            if None in blue_team_partial or None in red_team:
                continue
            
            try:
                payoff = calculate_full_payoff(blue_team_partial, red_team, blue_players, red_players)
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
                payoff = 0.0
            
            alpha = scorer.ALPHA
            U_blue = 1 / (1 + math.exp(-(payoff + alpha)))
            U_red = 1 - U_blue
            
            red_responses.append({
                'red_pick': red_pick,
                'payoff': payoff,
                'blue_winrate': U_blue,
                'red_winrate': U_red,
                'blue_team': blue_team_partial,
                'red_team': red_team
            })
            
            all_results.append({
                'blue_picks': blue_picks,
                'red_pick': red_pick,
                'payoff': payoff,
                'blue_winrate': U_blue,
                'red_winrate': U_red,
                'blue_team': blue_team_partial,
                'red_team': red_team
            })
        
        if red_responses:
            # Red's best response minimizes Blue's payoff (minimizes payoff)
            red_best = min(red_responses, key=lambda x: x['payoff'])
            
            blue_analysis.append({
                'blue_picks': blue_picks,
                'blue_combo_str': f"{blue_combo[0]} + {blue_combo[1]}",
                'red_best_response': red_best['red_pick'],
                'payoff_after_red_response': red_best['payoff'],
                'blue_winrate': red_best['blue_winrate'],
                'red_winrate': red_best['red_winrate'],
                'blue_team': red_best['blue_team'],
                'red_team': red_best['red_team']
            })
    
    # Sort by payoff after Red's best response (Blue wants to maximize)
    blue_analysis.sort(key=lambda x: x['payoff_after_red_response'], reverse=True)
    results = all_results  # For compatibility with later code
    
    if not blue_analysis:
        print("\n[ERROR] No valid combinations found!")
        return
    
    # Display backward induction results
    print(f"\n" + "="*70)
    print("BACKWARD INDUCTION RESULTS")
    print("="*70)
    print(f"\n   For each Blue combination, showing Red's BEST RESPONSE:")
    print(f"\n   {'#':<3} {'Blue Picks':<25} {'Red Response':<12} {'Payoff':>10} {'Blue WR':>10} {'Red WR':>10}")
    print("   " + "-" * 75)
    
    for i, analysis in enumerate(blue_analysis, 1):
        print(f"   {i:<3} {analysis['blue_combo_str']:<25} {analysis['red_best_response']:<12} {analysis['payoff_after_red_response']:>+10.4f} {analysis['blue_winrate']:>9.1%} {analysis['red_winrate']:>9.1%}")
    
    # SPNE: Blue picks the combination that maximizes payoff AFTER Red's best response
    spne_result = blue_analysis[0]  # Already sorted, first is best for Blue
    
    print(f"\n" + "="*70)
    print("SUBGAME PERFECT NASH EQUILIBRIUM (SPNE)")
    print("="*70)
    
    print(f"\n   Sequential Game Solution:")
    print(f"\n   Step 1: Blue picks knowing Red will respond optimally")
    print(f"   Step 2: Red observes Blue's picks, then picks best response")
    
    print(f"\n   BANNED ({len(draft.bans)}): {draft.bans}")
    
    role1, role2 = blue_open_roles
    print(f"\n   {TEAM_1['name']} (Blue) OPTIMAL picks:")
    print(f"      {role1}: {spne_result['blue_picks'][role1]}")
    print(f"      {role2}: {spne_result['blue_picks'][role2]}")
    
    print(f"\n   {TEAM_2['name']} (Red) BEST RESPONSE:")
    print(f"      {red_open_role}: {spne_result['red_best_response']}")
    
    # Show final teams by ROLE
    from get_worlds_data import ROLE_ORDER
    
    print(f"\n   Final Team Compositions (by role):")
    print(f"\n   {TEAM_1['name']} (Blue):")
    for i, role in enumerate(ROLE_ORDER):
        if i < len(spne_result['blue_team']):
            champ = spne_result['blue_team'][i]
            is_open = "(PICKED)" if role in blue_open_roles else ""
            print(f"      {role:<8}: {champ} {is_open}")
    
    print(f"\n   {TEAM_2['name']} (Red):")
    for i, role in enumerate(ROLE_ORDER):
        if i < len(spne_result['red_team']):
            champ = spne_result['red_team'][i]
            is_response = "(BEST RESPONSE)" if role == red_open_role else ""
            print(f"      {role:<8}: {champ} {is_response}")
    
    print(f"\n   SPNE Results:")
    print(f"   Payoff: {spne_result['payoff_after_red_response']:+.4f}")
    print(f"   {TEAM_1['name']} Win Probability: {spne_result['blue_winrate']:.1%}")
    print(f"   {TEAM_2['name']} Win Probability: {spne_result['red_winrate']:.1%}")
    
    # =========================================================================
    # LAST PICK IMPACT ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("LAST PICK IMPACT ANALYSIS")
    print("="*70)
    
    # Find the range of outcomes across all Blue combinations
    best_for_blue = blue_analysis[0]  # Already sorted best first
    worst_for_blue = blue_analysis[-1]
    
    # Win probabilities at extremes
    alpha = scorer.ALPHA
    
    print(f"\n   How much do Blue's 2 picks matter?")
    print(f"\n   If Blue picks OPTIMALLY ({best_for_blue['blue_combo_str']}):")
    print(f"      Red responds with: {best_for_blue['red_best_response']}")
    print(f"      Payoff: {best_for_blue['payoff_after_red_response']:+.4f} → Blue WR: {best_for_blue['blue_winrate']:.1%}")
    
    print(f"\n   If Blue picks WORST ({worst_for_blue['blue_combo_str']}):")
    print(f"      Red responds with: {worst_for_blue['red_best_response']}")
    print(f"      Payoff: {worst_for_blue['payoff_after_red_response']:+.4f} → Blue WR: {worst_for_blue['blue_winrate']:.1%}")
    
    swing = best_for_blue['blue_winrate'] - worst_for_blue['blue_winrate']
    print(f"\n   BLUE'S PICK IMPACT:")
    print(f"      Win rate swing: {swing*100:+.1f} percentage points")
    print(f"      (Blue's 2 picks can swing the game by {abs(swing)*100:.1f}%!)")
    
    # Red's impact: compare what happens if Red picks suboptimally
    print(f"\n   How much does Red's response matter?")
    # Get all Red options for the SPNE Blue combination
    spne_blue_picks = spne_result['blue_picks']
    red_options_for_spne = [r for r in all_results if r['blue_picks'] == spne_blue_picks]
    
    if len(red_options_for_spne) > 1:
        best_red = min(red_options_for_spne, key=lambda x: x['payoff'])  # Best for Red
        worst_red = max(red_options_for_spne, key=lambda x: x['payoff'])  # Worst for Red
        
        print(f"\n   Given Blue's optimal picks ({spne_result['blue_combo_str']}):")
        print(f"      If Red picks OPTIMALLY ({best_red['red_pick']}): Blue WR = {best_red['blue_winrate']:.1%}")
        print(f"      If Red picks WORST ({worst_red['red_pick']}): Blue WR = {worst_red['blue_winrate']:.1%}")
        
        red_swing = worst_red['blue_winrate'] - best_red['blue_winrate']
        print(f"\n   RED'S RESPONSE IMPACT:")
        print(f"      Win rate swing: {red_swing*100:+.1f} percentage points")
        print(f"      (Red's counter-pick can swing the game by {abs(red_swing)*100:.1f}%!)")
    
    # Show detailed breakdown for the SPNE outcome
    print("\n" + "="*70)
    print("DETAILED BREAKDOWN OF SPNE OUTCOME")
    print("="*70)
    
    log_full_payoff_breakdown(
        scorer, 
        spne_result['blue_team'], 
        spne_result['red_team'],
        TEAM_1['name'],
        TEAM_2['name'],
        pre_calculated_payoff=spne_result['payoff_after_red_response']
    )
    
    scorer.close()
    
    # Return results for PDF generation
    return {
        'draft': draft,
        'blue_open_roles': blue_open_roles,
        'red_open_role': red_open_role,
        'spne_result': spne_result,
        'blue_analysis': blue_analysis,
        'best_for_blue': blue_analysis[0],
        'worst_for_blue': blue_analysis[-1],
    }


def generate_pdf_report(all_results: List[Dict[str, Any]], filename: str = None):
    """Generate a PDF report of all draft analyses."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("[ERROR] fpdf2 not installed. Run: pip install fpdf2")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"draft_analysis_{timestamp}.pdf"
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "League of Legends Draft Analysis", ln=True, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 10, f"Iterations: {len(all_results)}", ln=True, align="C")
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Team 1: {TEAM_1['name']} (Blue Side)", ln=True)
    pdf.cell(0, 8, f"Team 2: {TEAM_2['name']} (Red Side)", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Champion Pool:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for role, champs in CHAMPION_POOL.items():
        pdf.cell(0, 6, f"  {role}: {', '.join(champs)}", ln=True)
    
    # Each iteration
    for i, result in enumerate(all_results, 1):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 15, f"Draft Scenario {i}", ln=True, align="C")
        pdf.ln(5)
        
        draft = result['draft']
        spne = result['spne_result']
        blue_roles = result['blue_open_roles']
        red_role = result['red_open_role']
        
        # Bans
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Banned Champions:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        bans_str = ", ".join(draft.bans)
        pdf.multi_cell(0, 6, bans_str)
        pdf.ln(5)
        
        # Draft State
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{TEAM_1['name']} (Blue) - Locked Picks:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        from get_worlds_data import ROLE_ORDER
        for role in ROLE_ORDER:
            champ = draft.team1_picks.get(role)
            if champ:
                pdf.cell(0, 6, f"  {role}: {champ}", ln=True)
            elif role in blue_roles:
                pdf.cell(0, 6, f"  {role}: (OPEN)", ln=True)
        
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{TEAM_2['name']} (Red) - Locked Picks:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for role in ROLE_ORDER:
            champ = draft.team2_picks.get(role)
            if champ:
                pdf.cell(0, 6, f"  {role}: {champ}", ln=True)
            elif role == red_role:
                pdf.cell(0, 6, f"  {role}: (OPEN)", ln=True)
        
        pdf.ln(5)
        
        # SPNE Result
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_fill_color(230, 230, 250)
        pdf.cell(0, 10, "SUBGAME PERFECT NASH EQUILIBRIUM", ln=True, fill=True, align="C")
        pdf.ln(3)
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, f"{TEAM_1['name']} (Blue) Optimal Picks:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        role1, role2 = blue_roles
        pdf.cell(0, 6, f"  {role1}: {spne['blue_picks'][role1]}", ln=True)
        pdf.cell(0, 6, f"  {role2}: {spne['blue_picks'][role2]}", ln=True)
        
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, f"{TEAM_2['name']} (Red) Best Response:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  {red_role}: {spne['red_best_response']}", ln=True)
        
        pdf.ln(5)
        
        # Win Probabilities
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Win Probabilities:", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"  {TEAM_1['name']}: {spne['blue_winrate']:.1%}", ln=True)
        pdf.cell(0, 7, f"  {TEAM_2['name']}: {spne['red_winrate']:.1%}", ln=True)
        pdf.cell(0, 7, f"  Payoff: {spne['payoff_after_red_response']:+.4f}", ln=True)
        
        pdf.ln(5)
        
        # Impact Analysis
        best = result['best_for_blue']
        worst = result['worst_for_blue']
        swing = best['blue_winrate'] - worst['blue_winrate']
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Pick Impact Analysis:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  Best Blue combo: {best['blue_combo_str']} -> {best['blue_winrate']:.1%} WR", ln=True)
        pdf.cell(0, 6, f"  Worst Blue combo: {worst['blue_combo_str']} -> {worst['blue_winrate']:.1%} WR", ln=True)
        pdf.cell(0, 6, f"  Win rate swing: {swing*100:+.1f} percentage points", ln=True)
        
        # Final Teams
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Final Team Compositions:", ln=True)
        
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(95, 6, f"{TEAM_1['name']} (Blue)", border=1, align="C")
        pdf.cell(95, 6, f"{TEAM_2['name']} (Red)", border=1, align="C", ln=True)
        
        pdf.set_font("Helvetica", "", 9)
        for j, role in enumerate(ROLE_ORDER):
            blue_champ = spne['blue_team'][j] if j < len(spne['blue_team']) else "-"
            red_champ = spne['red_team'][j] if j < len(spne['red_team']) else "-"
            pdf.cell(95, 5, f"{role}: {blue_champ}", border=1)
            pdf.cell(95, 5, f"{role}: {red_champ}", border=1, ln=True)
    
    # Summary page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 15, "Summary", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Results Across All Iterations:", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(20, 7, "#", border=1, align="C")
    pdf.cell(60, 7, "Blue Picks", border=1, align="C")
    pdf.cell(40, 7, "Red Response", border=1, align="C")
    pdf.cell(35, 7, "Blue WR", border=1, align="C")
    pdf.cell(35, 7, "Red WR", border=1, align="C", ln=True)
    
    for i, result in enumerate(all_results, 1):
        spne = result['spne_result']
        pdf.cell(20, 6, str(i), border=1, align="C")
        pdf.cell(60, 6, spne['blue_combo_str'], border=1, align="C")
        pdf.cell(40, 6, spne['red_best_response'], border=1, align="C")
        pdf.cell(35, 6, f"{spne['blue_winrate']:.1%}", border=1, align="C")
        pdf.cell(35, 6, f"{spne['red_winrate']:.1%}", border=1, align="C", ln=True)
    
    # Average win rates
    avg_blue_wr = sum(r['spne_result']['blue_winrate'] for r in all_results) / len(all_results)
    avg_red_wr = sum(r['spne_result']['red_winrate'] for r in all_results) / len(all_results)
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Average Win Rates Across {len(all_results)} Scenarios:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"  {TEAM_1['name']} (Blue): {avg_blue_wr:.1%}", ln=True)
    pdf.cell(0, 7, f"  {TEAM_2['name']} (Red): {avg_red_wr:.1%}", ln=True)
    
    # Save PDF
    pdf.output(filename)
    print(f"\n{'='*70}")
    print(f"PDF Report saved to: {filename}")
    print(f"{'='*70}")
    return filename


def run_multiple_analyses(num_iterations: int = 3, generate_pdf: bool = True):
    """Run multiple draft analyses with different random states."""
    print("="*70)
    print(f"RUNNING {num_iterations} DRAFT ANALYSES")
    print("="*70)
    
    all_results = []
    
    for i in range(num_iterations):
        print(f"\n{'#'*70}")
        print(f"# ITERATION {i+1} of {num_iterations}")
        print(f"{'#'*70}")
        
        result = test_worlds_draft_lastpick()
        if result:
            all_results.append(result)
        
        print(f"\n[ITERATION {i+1} COMPLETE]")
    
    if generate_pdf and all_results:
        generate_pdf_report(all_results)
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == 'sequential':
            test_spne_sequential_draft()
        elif test_type == 'partial':
            test_spne_partial_draft()
        elif test_type == 'simultaneous':
            test_spne_simultaneous()
        elif test_type == 'payoff':
            test_payoff_calculation()
        elif test_type == 'worlds' or test_type == 'lastpick':
            # Run 3 iterations and generate PDF
            run_multiple_analyses(num_iterations=3, generate_pdf=True)
        elif test_type == 'single':
            # Run single analysis (old behavior)
            test_worlds_draft_lastpick()
        else:
            print(f"Unknown test type: {test_type}")
            print("Available: sequential, partial, simultaneous, payoff, worlds/lastpick, single")
    else:
        # Run 3 iterations and generate PDF by default
        run_multiple_analyses(num_iterations=3, generate_pdf=True)

