"""
Test Game 1 scenario with Nash Equilibrium analysis.
Compares actual picks vs optimal SPNE strategy.
"""

from nash_equilibrium import SubgamePerfectNashEquilibrium
from composition_scorer import CompositionScorer
from test_worlds2025_scenarios import EXTENDED_CHAMPION_POOL, ALL_EXTENDED_CHAMPIONS, create_draft_from_scenario
from get_worlds_data import load_champion_name_map, normalize_champion_name, TEAM_1, TEAM_2, ROLE_ORDER, load_player_comfort_data
import math
import time
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress indicator
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, disable=False):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else None)
            self.desc = desc or ""
            self.unit = unit or "it"
            self.disable = disable
            self.n = 0
            self.start_time = time.time()
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if not self.disable:
                elapsed = time.time() - self.start_time
                print(f"\n{self.desc}: {self.n}/{self.total} completed in {elapsed:.2f}s")
        
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.update(1)
        
        def update(self, n=1):
            self.n += n
            if not self.disable and self.n % max(1, self.total // 20) == 0:
                elapsed = time.time() - self.start_time
                rate = self.n / elapsed if elapsed > 0 else 0
                pct = (self.n / self.total * 100) if self.total else 0
                print(f"\r{self.desc}: {self.n}/{self.total} ({pct:.1f}%) - {rate:.2f} {self.unit}/s", end='', flush=True)

# ============================================================================
# GAME 1 ACTUAL STATE
# ============================================================================
GAME_1_BANS = [
    "Bard", "Neeko", "Yunara", "Alistar", "Renata", 
    "Sivir", "Ezreal", "Yone", "Orianna", "Azir"
]

# KT (Blue Team 1) - 3 picks locked, 2 open (BOT and SUPPORT)
GAME_1_KT_LOCKED = {
    "TOP": "Rumble",
    "JUNGLE": "Wukong", 
    "MID": "Ryze",
    # BOT and SUPPORT are open
}

# T1 (Red Team 2) - 4 picks locked, 1 open (TOP)
GAME_1_T1_LOCKED = {
    "JUNGLE": "XinZhao",
    "MID": "Taliyah",
    "BOT": "Varus",
    "SUPPORT": "Poppy",
    # TOP is open
}

# Actual picks that were made
GAME_1_KT_ACTUAL = {
    "BOT": "Ashe",
    "SUPPORT": "Braum"
}

GAME_1_T1_ACTUAL = {
    "TOP": "Ambessa"
}


def normalize_champion_list(champ_list):
    """Normalize a list of champion names."""
    load_champion_name_map()
    return [normalize_champion_name(c) for c in champ_list if normalize_champion_name(c)]


def create_game1_draft_state():
    """Create draft state for Game 1."""
    load_champion_name_map()
    
    # Normalize bans
    normalized_bans = normalize_champion_list(GAME_1_BANS)
    
    # Normalize locked picks
    normalized_kt = {}
    for role, champ in GAME_1_KT_LOCKED.items():
        normalized = normalize_champion_name(champ)
        if normalized:
            normalized_kt[role] = normalized
    
    normalized_t1 = {}
    for role, champ in GAME_1_T1_LOCKED.items():
        normalized = normalize_champion_name(champ)
        if normalized:
            normalized_t1[role] = normalized
    
    # Create draft state
    draft = create_draft_from_scenario(
        bans=normalized_bans,
        team1_picks=normalized_kt,
        team1_last_role="BOT",  # Actually 2 open roles, but we'll handle separately
        team2_picks=normalized_t1,
        team2_last_role="TOP"
    )
    
    return draft


def run_nash_equilibrium_analysis():
    """Run Nash equilibrium analysis for Game 1."""
    print("=" * 70)
    print("GAME 1 NASH EQUILIBRIUM ANALYSIS")
    print("=" * 70)
    
    # Create draft state
    draft = create_game1_draft_state()
    
    print("\n" + "=" * 70)
    print("CURRENT DRAFT STATE")
    print("=" * 70)
    print(f"\nBans ({len(draft.bans)}): {', '.join(draft.bans)}")
    
    print(f"\n{TEAM_1['name']} (Blue) - Locked Picks:")
    for role in ROLE_ORDER:
        champ = draft.team1_picks.get(role)
        if champ:
            print(f"  {role:<8}: {champ}")
        else:
            print(f"  {role:<8}: [OPEN]")
    
    print(f"\n{TEAM_2['name']} (Red) - Locked Picks:")
    for role in ROLE_ORDER:
        champ = draft.team2_picks.get(role)
        if champ:
            print(f"  {role:<8}: {champ}")
        else:
            print(f"  {role:<8}: [OPEN]")
    
    # Get available champions
    all_picked = []
    for role in ROLE_ORDER:
        if draft.team1_picks.get(role):
            all_picked.append(draft.team1_picks[role])
        if draft.team2_picks.get(role):
            all_picked.append(draft.team2_picks[role])
    
    taken = set(draft.bans + all_picked)
    available = [c for c in ALL_EXTENDED_CHAMPIONS if c not in taken]
    
    print(f"\nAvailable Champions: {len(available)}")
    print(f"  {', '.join(available[:20])}{'...' if len(available) > 20 else ''}")
    
    # Initialize scorer and SPNE solver
    # ENABLE CACHING to avoid duplicate API calls
    print("=" * 70)
    print("INITIALIZING WITH API CALLS ENABLED")
    print("=" * 70)
    print("Note: Memoization is ENABLED to cache API calls and prevent duplicates")
    print("      This will speed up calculations significantly after the first run")
    print("")
    print("WARNING: API calls to u.gg can be slow (10s timeout per matchup)")
    print("         Each team matchup requires ~10 API calls (5 lanes × 2 directions)")
    print("         First run will take time, but results are cached for future runs")
    print("         If it hangs, press Ctrl+C and set skip_api_calls=True for faster testing")
    print("=" * 70)
    
    # Load player comfort data from database (fast, no API calls needed)
    print("\nLoading player comfort data from database...")
    load_player_comfort_data("worlds2025.db")
    print("Player comfort data loaded - will use database instead of OP.GG scraping")
    
    scorer = CompositionScorer()
    spne = SubgamePerfectNashEquilibrium(
        scorer,
        w1=0.15, w2=0.25, w3=0.6,
        use_memoization=True,  # ENABLED - cache API calls to avoid duplicates
        skip_api_calls=False,  # ENABLED - use full model with matchups and comfort
        # NOTE: Set to True for faster testing (skips API calls, uses composition only)
        show_progress=False,  # Disable built-in progress (we'll use tqdm)
        beam_width=1000,  # Very high beam width to prevent pruning (we only have ~224 Blue combinations)
        fast_heuristic=False  # Use full composition calculations for accuracy
    )
    
    # Debug flags disabled - calculations should be fast with caching
    spne._debug_timing = False
    scorer._debug_matchups = False
    
    # Get player names
    blue_players = [
        TEAM_1['players']['TOP'],
        TEAM_1['players']['JUNGLE'],
        TEAM_1['players']['MID'],
        TEAM_1['players']['BOT'],
        TEAM_1['players']['SUPPORT']
    ]
    red_players = [
        TEAM_2['players']['TOP'],
        TEAM_2['players']['JUNGLE'],
        TEAM_2['players']['MID'],
        TEAM_2['players']['BOT'],
        TEAM_2['players']['SUPPORT']
    ]
    
    # Convert draft state to team lists (with None for open slots)
    blue_team = [draft.team1_picks.get(role) for role in ROLE_ORDER]
    red_team = [draft.team2_picks.get(role) for role in ROLE_ORDER]
    
    print("\n" + "=" * 70)
    print("RUNNING NASH EQUILIBRIUM ANALYSIS")
    print("=" * 70)
    print("\nFinding optimal picks using backward induction...")
    print("(This may take a few minutes)\n")
    
    # Draft order: Blue picks 2 (BOT and SUPPORT), then Red picks 1 (TOP)
    draft_order = ['blue', 'blue', 'red']
    
    # Ensure caching is enabled (already enabled in constructor, but double-check)
    spne.use_memoization = True
    # Cache will accumulate entries to avoid duplicate API calls
    spne.state_cache.clear()
    spne.champion_data_cache.clear()
    spne.calculation_count = 0
    
    # Count how many TOP champions are available for Red's pick
    # Restrict to scenario's extended TOP pool
    top_champions = [c for c in EXTENDED_CHAMPION_POOL["TOP"] if c in available]
    
    # Count how many BOT and SUPPORT champions are available for Blue's picks
    # Restrict to scenario's extended BOT and SUPPORT pools
    bot_champions = [c for c in EXTENDED_CHAMPION_POOL["BOT"] if c in available]
    sup_champions = [c for c in EXTENDED_CHAMPION_POOL["SUPPORT"] if c in available]
    
    print(f"Starting fresh SPNE calculation...")
    print(f"  Beam width: {spne.beam_width} (set high to prevent pruning - we only have ~{len(bot_champions) * len(sup_champions)} Blue combinations)")
    print(f"  Fast heuristic: {spne.fast_heuristic}")
    print(f"  Memoization: {spne.use_memoization}")
    print(f"  Available champions: {len(available)}")
    print(f"  Max depth: 3 (Blue picks 2, Red picks 1)")
    print(f"  Available TOP champions for Red: {len(top_champions)}")
    print(f"  Available BOT champions for Blue: {len(bot_champions)}")
    print(f"  Available SUPPORT champions for Blue: {len(sup_champions)}")
    print(f"  Total combinations Blue can pick: {len(bot_champions) * len(sup_champions)}")
    print(f"  Expected calculations: ~{len(bot_champions) * len(sup_champions) * len(top_champions)} (all combinations)")
    print(f"  NOTE: With beam_width={spne.beam_width}, all {len(bot_champions) * len(sup_champions)} Blue combinations should be explored")
    
   
    
    print(f"  Top champions: {', '.join(top_champions[:10])}{'...' if len(top_champions) > 10 else ''}")
    
    try:
        # ========================================================================
        # STEP 1: Calculate Blue's optimal picks using minimax (brute force)
        # ========================================================================
        print("\n" + "=" * 70)
        print("STEP 1: BLUE'S OPTIMAL PICKS (USING MINIMAX)")
        print("=" * 70)
        
        # We'll calculate optimal picks in STEP 1.5 using minimax
        # For now, set placeholders that will be updated
        optimal_blue = blue_team.copy()
        optimal_red = red_team.copy()
        optimal_kt_bot = None
        optimal_kt_sup = None
        optimal_t1_top = None
        optimal_red_team = red_team.copy()
        
        print("Optimal picks will be determined in STEP 1.5 using minimax calculation...")
        
        # Define alpha for win rate calculations (Blue gets alpha bonus)
        alpha = 0.02
        
        # ========================================================================
        # STEP 1.5: Verify Blue's optimal picks by showing ALL combinations
        # ========================================================================
        print("\n" + "=" * 70)
        print("STEP 1.5: VERIFYING BLUE'S OPTIMAL PICKS")
        print("=" * 70)
        print(f"Testing ALL Blue combinations (BOT + SUPPORT) to verify optimal picks...\n")
        
        # Get all valid BOT and SUPPORT champions from scenario pools
        valid_bot_champions = []
        valid_sup_champions = []
        # BOT
        for champ in EXTENDED_CHAMPION_POOL["BOT"]:
            if champ in draft.bans or champ not in available:
                continue
            valid_bot_champions.append(champ)
        # SUPPORT
        for champ in EXTENDED_CHAMPION_POOL["SUPPORT"]:
            if champ in draft.bans or champ not in available:
                continue
            valid_sup_champions.append(champ)
        
        print(f"Found {len(valid_bot_champions)} valid BOT champions (scenario pool)")
        print(f"Found {len(valid_sup_champions)} valid SUPPORT champions (scenario pool)")
        print(f"Total combinations to test: {len(valid_bot_champions) * len(valid_sup_champions)}\n")
        
        # DON'T clear cache - reuse it to avoid duplicate API calls
        print(f"Cache status: {len(spne.payoff_cache)} entries (will reuse to avoid duplicate API calls)")
        
        # OPTIMIZATION: Pre-calculate locked role matchups to avoid redundant API calls
        print("\nPre-calculating locked role matchups...")
        locked_matchups = {}
        
        # Identify locked roles (roles where both teams have fixed picks)
        role_order = ['top', 'jungle', 'mid', 'adc', 'support']
        locked_roles = []
        for i, role in enumerate(role_order):
            blue_champ = blue_team[i] if i < len(blue_team) else None
            red_champ = red_team[i] if i < len(red_team) else None
            if blue_champ and red_champ:
                locked_roles.append((i, role, blue_champ, red_champ))
        
        # Pre-calculate matchups for locked roles
        # Clear entire cache to force fresh API calls with debug
        print("  Clearing matchup cache to force fresh API calls...")
        print(f"    Cache size before clearing: {len(spne.scorer.matchup_cache)}")
        spne.scorer.matchup_cache.clear()
        print(f"    Cleared entire matchup cache (now: {len(spne.scorer.matchup_cache)})")
        
        for i, role, blue_champ, red_champ in locked_roles:
            print(f"  Pre-calculating {role}: {blue_champ} vs {red_champ}...", end='', flush=True)
            # This will make the API call and cache it
            wr = spne.scorer.get_champion_matchup_winrate(blue_champ, red_champ, role)
            if wr is not None:
                print(f" {wr:.2%}")
            else:
                print(" (no data)")
            locked_matchups[(i, role)] = (blue_champ, red_champ, wr)
        
        print(f"Pre-calculated {len(locked_matchups)} locked role matchups")
        print(f"Matchup cache now has {len(spne.scorer.matchup_cache)} entries")
        
        # OPTIMIZATION: Pre-calculate ALL possible matchups that will be needed
        # For each Blue combination, we test all Red TOP picks
        # The only matchup that changes is: Rumble (Blue TOP) vs Red TOP champion
        # So we can pre-calculate Rumble vs all 25 possible Red TOP champions
        print("\nPre-calculating Rumble vs all possible Red TOP champions...")
        rumble_top_matchups = {}
        
        # Get all valid TOP champions for Red first
        # Check ALL champions in extended pool that can play TOP (including flex picks)
        all_picked = []
        for role in ROLE_ORDER:
            if draft.team1_picks.get(role):
                all_picked.append(draft.team1_picks[role])
            if draft.team2_picks.get(role):
                all_picked.append(draft.team2_picks[role])
        
        taken = set(draft.bans + all_picked)
        valid_top_champions = []
        seen_champs = set()  # Track to avoid duplicates
        # Check only champions in scenario's extended TOP pool
        for champ in EXTENDED_CHAMPION_POOL["TOP"]:
            if champ in taken or champ in seen_champs:
                continue
            valid_top_champions.append(champ)
            seen_champs.add(champ)
        
        print(f"Found {len(valid_top_champions)} valid TOP champions for Red (scenario pool)")
        
        # OPTIMIZATION: Pre-calculate Rumble vs all Red TOP champions
        # This is the ONLY matchup that varies across calculations
        # Without this, we'd make 222 × 25 = 5,550 API calls for the same matchups!
        rumble_champ = blue_team[0]  # Rumble is Blue's locked TOP
        print(f"\nPre-calculating {rumble_champ} vs all {len(valid_top_champions)} Red TOP champions...")
        print(f"  This will make {len(valid_top_champions)} API calls upfront, but save {222 * len(valid_top_champions) - len(valid_top_champions)} redundant calls later")
        
        rumble_top_matchups = {}
        successful_rumble_matchups = 0
        with tqdm(total=len(valid_top_champions), desc="Pre-calc Rumble matchups", unit="champ") as pbar:
            for top_champ in sorted(valid_top_champions):
                # Get Rumble vs TOP winrate (this populates the cache)
                wr = spne.scorer.get_champion_matchup_winrate(rumble_champ, top_champ, 'top')
                rumble_top_matchups[top_champ] = wr
                if wr is not None:
                    successful_rumble_matchups += 1
                pbar.update(1)
                if wr is not None:
                    pbar.set_postfix({'last': f"{top_champ}: {wr:.2%}", 'cache': len(spne.scorer.matchup_cache)})
        
        print(f"Pre-calculated {successful_rumble_matchups}/{len(valid_top_champions)} Rumble vs TOP matchups (cache now has {len(spne.scorer.matchup_cache)} entries)")
        
        # OPTIMIZATION: Pre-calculate Varus vs all Blue ADC champions
        # We'll invert the winrate when we need ADC vs Varus
        varus_champ = red_team[3]  # Varus is Red's locked BOT
        print(f"\nPre-calculating {varus_champ} vs all {len(valid_bot_champions)} Blue ADC champions...")
        print(f"  This will make {len(valid_bot_champions)} API calls, then we'll invert for ADC vs Varus")
        
        varus_vs_adc_matchups = {}
        successful_varus_matchups = 0
        with tqdm(total=len(valid_bot_champions), desc="Pre-calc Varus vs ADC", unit="champ") as pbar:
            for adc_champ in sorted(valid_bot_champions):
                # Get Varus vs ADC winrate (this populates the cache)
                wr = spne.scorer.get_champion_matchup_winrate(varus_champ, adc_champ, 'adc')
                varus_vs_adc_matchups[adc_champ] = wr
                if wr is not None:
                    successful_varus_matchups += 1
                pbar.update(1)
                if wr is not None:
                    pbar.set_postfix({'last': f"{adc_champ}: {wr:.2%}", 'cache': len(spne.scorer.matchup_cache)})
        
        print(f"Pre-calculated {successful_varus_matchups}/{len(valid_bot_champions)} Varus vs ADC matchups (cache now has {len(spne.scorer.matchup_cache)} entries)")
        
        # OPTIMIZATION: Pre-calculate Poppy vs all Blue SUPPORT champions
        # The cache automatically handles inversion - when we need SUPPORT vs Poppy,
        # it will use the cached value (no extra API calls or manual inversion needed)
        poppy_champ = red_team[4]  # Poppy is Red's locked SUPPORT
        print(f"\nPre-calculating {poppy_champ} vs all {len(valid_sup_champions)} Blue SUPPORT champions...")
        print(f"  Cache will automatically handle SUPPORT vs Poppy (no manual inversion needed)")
        
        poppy_vs_sup_matchups = {}
        successful_poppy_matchups = 0
        with tqdm(total=len(valid_sup_champions), desc="Pre-calc Poppy vs SUPPORT", unit="champ") as pbar:
            for sup_champ in sorted(valid_sup_champions):
                # Get Poppy vs SUPPORT winrate (this populates the cache)
                wr = spne.scorer.get_champion_matchup_winrate(poppy_champ, sup_champ, 'support')
                poppy_vs_sup_matchups[sup_champ] = wr
                if wr is not None:
                    successful_poppy_matchups += 1
                pbar.update(1)
                if wr is not None:
                    pbar.set_postfix({'last': f"{sup_champ}: {wr:.2%}", 'cache': len(spne.scorer.matchup_cache)})
        
        print(f"Pre-calculated {successful_poppy_matchups}/{len(valid_sup_champions)} Poppy vs SUPPORT matchups (cache now has {len(spne.scorer.matchup_cache)} entries)")
        print(f"Matchup cache now has {len(spne.scorer.matchup_cache)} entries")
        
        # VERIFY: Test that cache is working for a few matchups
        print(f"\nVerifying cache integration...")
        test_cache_hits = 0
        test_cache_misses = 0
        
        # Test Rumble vs TOP (should be cached)
        test_top = valid_top_champions[0] if valid_top_champions else None
        if test_top:
            cache_before = len(spne.scorer.matchup_cache)
            wr_test = spne.scorer.get_champion_matchup_winrate(rumble_champ, test_top, 'top')
            cache_after = len(spne.scorer.matchup_cache)
            if cache_after == cache_before:
                test_cache_hits += 1
                print(f"  ✓ Rumble vs {test_top} (TOP): CACHE HIT")
            else:
                test_cache_misses += 1
                print(f"  ✗ Rumble vs {test_top} (TOP): CACHE MISS (unexpected!)")
        
        # Test ADC vs Varus (should be cached via inversion)
        test_adc = valid_bot_champions[0] if valid_bot_champions else None
        if test_adc:
            cache_before = len(spne.scorer.matchup_cache)
            wr_test = spne.scorer.get_champion_matchup_winrate(test_adc, varus_champ, 'adc')
            cache_after = len(spne.scorer.matchup_cache)
            if cache_after == cache_before:
                test_cache_hits += 1
                print(f"  ✓ {test_adc} vs Varus (ADC): CACHE HIT (inverted)")
            else:
                test_cache_misses += 1
                print(f"  ✗ {test_adc} vs Varus (ADC): CACHE MISS (unexpected!)")
        
        # Test SUPPORT vs Poppy (should be cached via inversion)
        test_sup = valid_sup_champions[0] if valid_sup_champions else None
        if test_sup:
            cache_before = len(spne.scorer.matchup_cache)
            wr_test = spne.scorer.get_champion_matchup_winrate(test_sup, poppy_champ, 'support')
            cache_after = len(spne.scorer.matchup_cache)
            if cache_after == cache_before:
                test_cache_hits += 1
                print(f"  ✓ {test_sup} vs Poppy (SUPPORT): CACHE HIT (inverted)")
            else:
                test_cache_misses += 1
                print(f"  ✗ {test_sup} vs Poppy (SUPPORT): CACHE MISS (unexpected!)")
        
        if test_cache_misses == 0:
            print(f"  ✓ All pre-calculated matchups are properly cached and will be reused!")
        else:
            print(f"  ⚠ Warning: {test_cache_misses} cache miss(es) detected - some matchups may not be cached correctly")
        
        print(f"\nTotal calculations needed: {len(valid_bot_champions) * len(valid_sup_champions) * len(valid_top_champions)}")
        print(f"  ({len(valid_bot_champions)} BOT × {len(valid_sup_champions)} SUPPORT × {len(valid_top_champions)} TOP)")
        print(f"  Expected cache hits: All Rumble vs TOP, all ADC vs Varus (inverted), all SUPPORT vs Poppy (inverted)")
        print(f"  Expected new API calls: Only for any new matchups not pre-calculated\n")
        
        blue_combinations = []
        
        print("=" * 70)
        print("ALL BLUE COMBINATIONS (BOT + SUPPORT):")
        print("=" * 70)
        
        # Calculate total combinations
        total_combos = 0
        for bot_champ in sorted(valid_bot_champions):
            for sup_champ in sorted(valid_sup_champions):
                if bot_champ != sup_champ:
                    total_combos += 1
        
        print(f"Calculating {total_combos} Blue combinations × {len(valid_top_champions)} Red TOP picks...")
        print(f"Total calculations: {total_combos * len(valid_top_champions)}")
        print(f"Cache will prevent duplicate API calls for same team matchups\n")
        
        combo_count = 0
        seen_combos = set()  # Track combinations to avoid duplicates
        
        # Progress bar for Blue combinations
        bot_sup_combos = []
        for bot_champ in sorted(valid_bot_champions):
            for sup_champ in sorted(valid_sup_champions):
                if bot_champ != sup_champ:
                    bot_sup_combos.append((bot_champ, sup_champ))
        
        # Calculate all Blue combinations (no progress bar - should be fast with caching)
        for bot_champ, sup_champ in bot_sup_combos:
            # Check for duplicate combinations
            combo_key = (bot_champ, sup_champ)
            if combo_key in seen_combos:
                continue
            seen_combos.add(combo_key)
            
            combo_count += 1
            test_blue = blue_team.copy()
            test_blue[3] = bot_champ  # BOT
            test_blue[4] = sup_champ  # SUPPORT
            
            # For each Blue combination, test ALL Red TOP picks and find the minimum payoff
            # (Red will pick the TOP that minimizes Blue's payoff)
            best_red_top_for_combo = None
            best_payoff_for_combo = float('inf')  # Red wants minimum (most negative)
            all_red_responses = []  # Track all Red responses for this Blue combo
            
            # Test ALL Red TOP picks for this Blue combination
            for top_champ in sorted(valid_top_champions):
                # Can't pick same champion that Blue already has
                if top_champ in test_blue:
                    continue
                
                # Double-check: make sure champion isn't None or already picked
                if not top_champ or top_champ in draft.bans:
                    continue
                
                test_red = red_team.copy()
                test_red[0] = top_champ
                
                test_payoff = spne.calculate_payoff(
                    test_blue, test_red,
                    blue_players, red_players, 'na1', use_fast_mode=False
                )
                
                all_red_responses.append((top_champ, test_payoff))
                
                # Red wants the MINIMUM payoff (most negative)
                if test_payoff < best_payoff_for_combo:
                    best_payoff_for_combo = test_payoff
                    best_red_top_for_combo = top_champ
            
            # The payoff for this Blue combination is the minimum (Red's best response)
            test_payoff = best_payoff_for_combo
            
            # Blue gets alpha bonus
            kt_win = 1 / (1 + math.exp(-(test_payoff + alpha)))
            blue_combinations.append((bot_champ, sup_champ, test_payoff, kt_win, best_red_top_for_combo))
        
        # Sort by payoff (descending - highest first = best for Blue)
        blue_combinations.sort(key=lambda x: x[2], reverse=True)
        
        # VERIFY: The minimax best combination should match SPNE
        minimax_best = blue_combinations[0]  # Highest payoff
        minimax_bot, minimax_sup, minimax_payoff, minimax_win, minimax_red_top = minimax_best
        
        # Now print the table with SPNE marked
        print(f"\n{'BOT':<20} {'SUPPORT':<20} {'Payoff':>12} {'KT Win%':>10} {'Red Best TOP':<20} {'Status':<20}")
        print("-" * 90)
        
        for bot_champ, sup_champ, test_payoff, kt_win, best_red_top_for_combo in blue_combinations:
            # Determine status (mark SPNE - the one with highest payoff)
            status = ""
            if bot_champ == minimax_bot and sup_champ == minimax_sup:
                status = " [SPNE]"
            
            # Format payoff
            payoff_str = f"{test_payoff:>+12.4f}"
            
            red_top_str = best_red_top_for_combo if best_red_top_for_combo else "N/A"
            print(f"{bot_champ:<20} {sup_champ:<20} {payoff_str:>30} {kt_win:>9.2%} {red_top_str:<20} {status}")
        
        print(f"\n{'='*70}")
        print("STEP 1 RESULT: SPNE")
        print("=" * 70)
        print(f"SPNE (tests ALL {len(blue_combinations)} combinations):")
        print(f"  Optimal Blue picks: BOT={minimax_bot}, SUPPORT={minimax_sup}, Payoff={minimax_payoff:+.4f} [SPNE]")
        print(f"  Red's best response: {minimax_red_top}")
        print(f"  KT Win Rate: {minimax_win:.2%}")
        print(f"{'='*70}")
        
        # Update optimal picks to use minimax result
        optimal_kt_bot = minimax_bot
        optimal_kt_sup = minimax_sup
        optimal_blue[3] = minimax_bot
        optimal_blue[4] = minimax_sup
        optimal_red[0] = minimax_red_top
        optimal_red_team[0] = minimax_red_top
        optimal_t1_top = minimax_red_top
        
        print(f"\n{TEAM_1['name']} (Blue) - Optimal Picks (from Minimax):")
        for i, role in enumerate(ROLE_ORDER):
            champ = optimal_blue[i] if i < len(optimal_blue) else None
            is_new = champ and champ != draft.team1_picks.get(role)
            marker = " [OPTIMAL]" if is_new else ""
            print(f"  {role:<8}: {champ}{marker}")
        
        print(f"\n{TEAM_2['name']} (Red) - Optimal Response (from Minimax):")
        for i, role in enumerate(ROLE_ORDER):
            champ = optimal_red[i] if i < len(optimal_red) else None
            is_new = champ and champ != draft.team2_picks.get(role)
            marker = " [OPTIMAL]" if is_new else ""
            print(f"  {role:<8}: {champ}{marker}")
        
        # Create result dictionary from minimax calculation
        result = {
            'blue_team': optimal_blue,
            'red_team': optimal_red,
            'payoff': minimax_payoff,
            'strategy': None,
            'best_response': None
        }
        
        # Count how many times each TOP champion was selected as best response
        top_champ_counts = {}
        for combo in blue_combinations:
            if len(combo) >= 5:
                red_top = combo[4]
                top_champ_counts[red_top] = top_champ_counts.get(red_top, 0) + 1
        
        print("\n" + "=" * 70)
        print("SUMMARY - TOP 10 BEST COMBINATIONS FOR BLUE:")
        print("=" * 70)
        for i, combo in enumerate(blue_combinations[:10]):
            bot, sup, payoff, win, red_top = combo
            marker = " [SPNE]" if bot == optimal_kt_bot and sup == optimal_kt_sup else ""
            advantage = "Blue advantage" if payoff > 0 else "Red advantage"
            print(f"  {i+1:2d}. BOT={bot:<15} SUP={sup:<15} Payoff={payoff:>+10.4f} ({advantage}), KT Win={win:>6.2%}, Red TOP={red_top}{marker}")
        
        print(f"\n{'='*70}")
        print(f"BLUE'S BEST COMBINATION (SPNE): BOT={minimax_bot}, SUPPORT={minimax_sup}")
        print(f"  Red's optimal TOP response: {minimax_red_top}")
        print(f"  Payoff: {minimax_payoff:+.4f}")
        print(f"  KT Win Rate: {minimax_win:.2%}")
        print(f"{'='*70}")
        print(f"\nTotal calculations for Blue combinations: {spne.calculation_count:,}")
        print(f"  Formula: {len(valid_bot_champions)} BOT × {len(valid_sup_champions)} SUPPORT × {len(valid_top_champions)} TOP = {len(valid_bot_champions) * len(valid_sup_champions) * len(valid_top_champions)} total")
        
        # ========================================================================
        # STEP 2: Get Blue's actual picks
        # ========================================================================
        print("\n" + "=" * 70)
        print("STEP 2: BLUE'S ACTUAL PICKS")
        print("=" * 70)
        
        # Normalize actual picks
        load_champion_name_map()
        kt_actual_bot = normalize_champion_name(GAME_1_KT_ACTUAL["BOT"])
        kt_actual_sup = normalize_champion_name(GAME_1_KT_ACTUAL["SUPPORT"])
        t1_actual_top = normalize_champion_name(GAME_1_T1_ACTUAL["TOP"])
        
        actual_blue_team = blue_team.copy()
        actual_blue_team[3] = kt_actual_bot  # BOT
        actual_blue_team[4] = kt_actual_sup  # SUPPORT
        
        print(f"\n{TEAM_1['name']} (Blue) - Actual Picks:")
        print(f"  BOT:     {kt_actual_bot}")
        print(f"  SUPPORT: {kt_actual_sup}")
        
        # Build team combinations
        optimal_blue_team = optimal_blue.copy()
        optimal_red_team = optimal_red.copy()
        
        actual_red_team = red_team.copy()
        actual_red_team[0] = t1_actual_top  # TOP
        
        alpha = 0.02
        
        # Calculate all 4 scenarios
        print("\n" + "=" * 70)
        print("SCENARIO COMPARISONS")
        print("=" * 70)
        
        scenarios = [
            (f"Optimal ({optimal_kt_bot}, {optimal_kt_sup}) vs Optimal ({optimal_t1_top})", optimal_blue_team, optimal_red_team),
            (f"Actual ({kt_actual_bot}, {kt_actual_sup}) vs Actual ({t1_actual_top})", actual_blue_team, actual_red_team),
            (f"Optimal ({optimal_kt_bot}, {optimal_kt_sup}) vs Actual ({t1_actual_top})", optimal_blue_team, actual_red_team),
            (f"Actual ({kt_actual_bot}, {kt_actual_sup}) vs Optimal ({optimal_t1_top})", actual_blue_team, optimal_red_team),
        ]
        
        results = []
        for name, blue, red in scenarios:
            payoff = spne.calculate_payoff(
                blue, red,
                blue_players, red_players, 'na1', use_fast_mode=True
            )
            blue_win = 1 / (1 + math.exp(-(payoff + alpha)))
            red_win = 1 - blue_win
            results.append((name, payoff, blue_win, red_win))
        
        print(f"\n{'Scenario':<25} {'Payoff':>10} {'KT Win%':>10} {'T1 Win%':>10}")
        print("-" * 60)
        for name, payoff, blue_win, red_win in results:
            print(f"{name:<25} {payoff:>+10.4f} {blue_win:>9.1%} {red_win:>9.1%}")
        
        # Detailed breakdown
        print("\n" + "=" * 70)
        print("DETAILED COMPARISON")
        print("=" * 70)
        
        print(f"\n{TEAM_1['name']} (Blue) Last Picks:")
        print(f"  BOT:     Actual={kt_actual_bot}, Optimal={optimal_kt_bot}, Match={kt_actual_bot == optimal_kt_bot}")
        print(f"  SUPPORT: Actual={kt_actual_sup}, Optimal={optimal_kt_sup}, Match={kt_actual_sup == optimal_kt_sup}")
        
        print(f"\n{TEAM_2['name']} (Red) Last Pick:")
        print(f"  TOP:     Actual={t1_actual_top}, Optimal={optimal_t1_top}, Match={t1_actual_top == optimal_t1_top}")
        
        # Analyze each scenario
        opt_vs_opt = results[0]
        actual_vs_actual = results[1]
        opt_vs_actual = results[2]
        actual_vs_opt = results[3]
        
        print(f"\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        
        print(f"\n1. Optimal ({optimal_kt_bot}, {optimal_kt_sup}) vs Optimal ({optimal_t1_top}) (SPNE):")
        print(f"   Payoff: {opt_vs_opt[1]:+.4f}, KT: {opt_vs_opt[2]:.1%}, T1: {opt_vs_opt[3]:.1%}")
        
        print(f"\n2. Actual ({kt_actual_bot}, {kt_actual_sup}) vs Actual ({t1_actual_top}) (What happened):")
        print(f"   Payoff: {actual_vs_actual[1]:+.4f}, KT: {actual_vs_actual[2]:.1%}, T1: {actual_vs_actual[3]:.1%}")
        diff_actual = actual_vs_actual[1] - opt_vs_opt[1]
        if abs(diff_actual) < 0.01:
            print(f"   -> Actual outcome matches optimal outcome!")
        elif diff_actual > 0:
            print(f"   -> Actual outcome is {diff_actual:+.4f} better for KT than optimal")
        else:
            print(f"   -> Optimal would be {abs(diff_actual):+.4f} better for KT than actual")
        
        print(f"\n3. Optimal ({optimal_kt_bot}, {optimal_kt_sup}) vs Actual ({t1_actual_top}) (If Blue optimal, Red actual):")
        print(f"   Payoff: {opt_vs_actual[1]:+.4f}, KT: {opt_vs_actual[2]:.1%}, T1: {opt_vs_actual[3]:.1%}")
        print(f"   -> Shows value of Blue's optimal picks when Red doesn't respond optimally")
        
        print(f"\n4. Actual ({kt_actual_bot}, {kt_actual_sup}) vs Optimal ({optimal_t1_top}) (If Blue actual, Red optimal response):")
        print(f"   Payoff: {actual_vs_opt[1]:+.4f}, KT: {actual_vs_opt[2]:.1%}, T1: {actual_vs_opt[3]:.1%}")
        print(f"   -> Shows how Red's optimal response performs against Blue's actual picks")
        
        # ========================================================================
        # STEP 3: Calculate Red's best response to Blue's optimal picks
        # ========================================================================
        print("\n" + "=" * 70)
        print("STEP 3: RED'S BEST RESPONSE TO BLUE'S OPTIMAL PICKS")
        print("=" * 70)
        
        # DON'T clear cache - reuse it to avoid duplicate API calls
        # Cache is keyed by team composition, so it will automatically reuse previous calculations
        print(f"Cache status: {len(spne.payoff_cache)} entries (will reuse to avoid duplicate API calls)")
        
        t1_best_response_to_optimal = None
        best_t1_payoff_to_optimal = float('inf')  # Red wants MINIMUM payoff (most negative = best for Red)
        t1_responses_to_optimal = []  # Track all responses for analysis
        
        # Get all valid TOP champions (no duplicates, not in Blue team) from scenario pool
        valid_top_champions = []
        seen_champs = set()  # Track to avoid duplicates
        for champ in EXTENDED_CHAMPION_POOL["TOP"]:
            if champ in draft.bans or champ in optimal_blue_team or champ in seen_champs:
                continue
            valid_top_champions.append(champ)
            seen_champs.add(champ)
        
        print("=" * 70)
        print("ALL CALCULATIONS:")
        print("=" * 70)
        print(f"{'Champion':<20} {'Payoff':>12} {'T1 Win%':>10} {'Status':<20}")
        print("-" * 70)
        
        # Calculate Red TOP picks (no progress bar - should be fast with caching)
        for champ in sorted(valid_top_champions):  # Sort alphabetically for consistent output
            test_red = optimal_red_team.copy()
            test_red[0] = champ
            # Cache will prevent duplicate API calls for same team matchups
            test_payoff = spne.calculate_payoff(
                optimal_blue_team, test_red,
                blue_players, red_players, 'na1', use_fast_mode=False
            )
            
            # Calculate win rate: payoff is from Blue's perspective
            # Blue gets alpha bonus, so Blue's win rate = 1 / (1 + exp(-(payoff + alpha)))
            # Red's win rate = 1 - Blue's win rate = 1 / (1 + exp(payoff + alpha))
            blue_win = 1 / (1 + math.exp(-(test_payoff + alpha)))
            t1_win = 1 - blue_win  # Red's win rate (no alpha bonus for Red)
            t1_responses_to_optimal.append((champ, test_payoff, t1_win))
            
            # Determine status (only mark SPNE)
            status = ""
            if test_payoff < best_t1_payoff_to_optimal:
                best_t1_payoff_to_optimal = test_payoff
                t1_best_response_to_optimal = champ
            if champ == optimal_t1_top:
                status = " [SPNE]"
            
            payoff_str = f"{test_payoff:>+12.4f}"
            
            # Show win rate with more precision to see differences
            print(f"{champ:<20} {payoff_str:>30} {t1_win:>9.2%} {status}")
        
        # Sort by payoff (ascending - most negative first = best for Red)
        t1_responses_to_optimal.sort(key=lambda x: x[1])
        
        print(f"\n{'='*70}")
        print(f"RED'S BEST RESPONSE: {t1_best_response_to_optimal}")
        print(f"  Payoff: {best_t1_payoff_to_optimal:+.4f} ({'Red advantage' if best_t1_payoff_to_optimal < 0 else 'Blue advantage'})")
        # Blue gets alpha bonus, Red does not
        blue_win_best = 1 / (1 + math.exp(-(best_t1_payoff_to_optimal + alpha)))
        t1_win_best = 1 - blue_win_best
        print(f"  T1 Win Rate: {t1_win_best:.2%}")
        if t1_best_response_to_optimal != optimal_t1_top:
            minimax_response = next((p for c, p, w in t1_responses_to_optimal if c == optimal_t1_top), None)
            if minimax_response is not None:
                minimax_payoff_red = minimax_response
                print(f"  Minimax pick was: {optimal_t1_top} (payoff: {minimax_payoff_red:+.4f})")
                print(f"  Difference: {best_t1_payoff_to_optimal - minimax_payoff_red:+.4f} (Red should pick the MORE NEGATIVE payoff)")
        print(f"{'='*70}")
        print(f"\nTotal calculations performed: {spne.calculation_count:,}")
        
        # ========================================================================
        # STEP 4: Calculate Red's best response to Blue's actual picks
        # ========================================================================
        print("\n" + "=" * 70)
        print("STEP 4: RED'S BEST RESPONSE TO BLUE'S ACTUAL PICKS")
        print("=" * 70)
        # DON'T clear cache - reuse it to avoid duplicate API calls
        print(f"Cache status: {len(spne.payoff_cache)} entries (will reuse to avoid duplicate API calls)")
        
        t1_best_response_to_actual = None
        best_t1_payoff_to_actual = float('inf')  # Red wants to minimize Blue's payoff
        t1_responses_to_actual = []  # Track all responses for analysis
        
        # Get all valid TOP champions from scenario pool
        valid_top_champions_actual = []
        seen_champs = set()  # Track to avoid duplicates
        for champ in EXTENDED_CHAMPION_POOL["TOP"]:
            if champ in draft.bans or champ in actual_blue_team or champ in seen_champs:
                continue
            valid_top_champions_actual.append(champ)
            seen_champs.add(champ)
        
        # Calculate Red TOP picks (actual) - no progress bar, should be fast with caching
        for champ in sorted(valid_top_champions_actual):
            test_red = actual_red_team.copy()
            test_red[0] = champ
            # Cache will prevent duplicate API calls for same team matchups
            test_payoff = spne.calculate_payoff(
                actual_blue_team, test_red,
                blue_players, red_players, 'na1', use_fast_mode=False
            )
            
            # Blue gets alpha bonus, Red does not
            blue_win = 1 / (1 + math.exp(-(test_payoff + alpha)))
            t1_win = 1 - blue_win  # Red's win rate
            t1_responses_to_actual.append((champ, test_payoff, t1_win))
            
            if test_payoff < best_t1_payoff_to_actual:
                best_t1_payoff_to_actual = test_payoff
                t1_best_response_to_actual = champ
        
        # Sort and show top responses
        t1_responses_to_actual.sort(key=lambda x: x[1])  # Sort by payoff (lower = better for Red)
        print(f"Top 5 responses:")
        for i, (champ, payoff, win) in enumerate(t1_responses_to_actual[:5]):
            marker = " [BEST]" if i == 0 else " [ACTUAL]" if champ == t1_actual_top else ""
            print(f"  {i+1}. {champ}: Payoff={payoff:+.4f}, T1 Win={win:.1%}{marker}")
        
        print(f"\nCalculations performed: {spne.calculation_count:,}")
        
        # ========================================================================
        # STEP 5: Compare all scenarios
        # ========================================================================
        print("\n" + "=" * 70)
        print("STEP 5: COMPARISON OF ALL SCENARIOS")
        print("=" * 70)
        
        # Build teams for each scenario
        # 1. Optimal Blue vs Best Response to Optimal
        optimal_vs_br_optimal_red = optimal_red_team.copy()
        optimal_vs_br_optimal_red[0] = t1_best_response_to_optimal
        
        # 2. Optimal Blue vs SPNE Red
        optimal_vs_spne_red = optimal_red_team.copy()
        
        # 3. Actual Blue vs Best Response to Actual
        actual_vs_br_actual_red = actual_red_team.copy()
        actual_vs_br_actual_red[0] = t1_best_response_to_actual
        
        # 4. Actual Blue vs Actual Red
        actual_vs_actual_red = actual_red_team.copy()
        
        # Cache will be reused to avoid duplicate API calls
        print(f"Cache status: {len(spne.payoff_cache)} entries (will reuse to avoid duplicate API calls)")
        
        scenarios = [
            (f"Optimal Blue ({optimal_kt_bot}, {optimal_kt_sup}) vs SPNE Red ({optimal_t1_top})", optimal_blue_team, optimal_vs_spne_red),
            (f"Optimal Blue ({optimal_kt_bot}, {optimal_kt_sup}) vs Best Response ({t1_best_response_to_optimal})", optimal_blue_team, optimal_vs_br_optimal_red),
            (f"Actual Blue ({kt_actual_bot}, {kt_actual_sup}) vs Best Response ({t1_best_response_to_actual})", actual_blue_team, actual_vs_br_actual_red),
            (f"Actual Blue ({kt_actual_bot}, {kt_actual_sup}) vs Actual Red ({t1_actual_top})", actual_blue_team, actual_vs_actual_red),
        ]
        
        results = []
        for name, blue, red in scenarios:
            payoff = spne.calculate_payoff(
                blue, red,
                blue_players, red_players, 'na1', use_fast_mode=False
            )
            blue_win = 1 / (1 + math.exp(-(payoff + alpha)))
            red_win = 1 - blue_win
            results.append((name, payoff, blue_win, red_win))
        
        print(f"\n{'Scenario':<40} {'Payoff':>10} {'KT Win%':>10} {'T1 Win%':>10}")
        print("-" * 75)
        for name, payoff, blue_win, red_win in results:
            print(f"{name:<40} {payoff:>+10.4f} {blue_win:>9.1%} {red_win:>9.1%}")
        
        print(f"\nTotal calculations for scenarios: {spne.calculation_count:,}")
        
        # Extract results
        opt_vs_spne = results[0]
        opt_vs_br_opt = results[1]
        actual_vs_br_actual = results[2]
        actual_vs_actual = results[3]
        
        print(f"\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)
        
        print(f"\n1. Blue's Optimal Picks ({optimal_kt_bot}, {optimal_kt_sup}): BOT={optimal_kt_bot}, SUPPORT={optimal_kt_sup}")
        print(f"   Red's SPNE response ({optimal_t1_top}): {optimal_t1_top}")
        print(f"     -> Payoff: {opt_vs_spne[1]:+.4f} (T1 win: {opt_vs_spne[3]:.1%})")
        print(f"   Red's best response ({t1_best_response_to_optimal}): {t1_best_response_to_optimal}")
        print(f"     -> Payoff: {opt_vs_br_opt[1]:+.4f} (T1 win: {opt_vs_br_opt[3]:.1%})")
        
        if t1_best_response_to_optimal == optimal_t1_top:
            print(f"   [OK] SPNE pick ({optimal_t1_top}) matches best response ({t1_best_response_to_optimal})!")
        else:
            # Red wants the MORE NEGATIVE payoff (lower value)
            payoff_diff = opt_vs_br_opt[1] - opt_vs_spne[1]  # Negative means best response is better for Red
            win_diff = opt_vs_br_opt[3] - opt_vs_spne[3]
            print(f"   [WARNING] SPNE pick ({optimal_t1_top}) is NOT the best response ({t1_best_response_to_optimal})!")
            print(f"   Best response payoff ({opt_vs_br_opt[1]:+.4f}) is {abs(payoff_diff):.4f} {'more negative' if payoff_diff < 0 else 'less negative'} than SPNE ({opt_vs_spne[1]:+.4f})")
            print(f"   This gives T1 {win_diff:+.1%} better win rate")
        
        print(f"\n2. Blue's Actual Picks ({kt_actual_bot}, {kt_actual_sup}): BOT={kt_actual_bot}, SUPPORT={kt_actual_sup}")
        print(f"   Red's best response ({t1_best_response_to_actual}): {t1_best_response_to_actual}")
        print(f"     -> Payoff: {actual_vs_br_actual[1]:+.4f} (T1 win: {actual_vs_br_actual[3]:.1%})")
        print(f"   Red's actual pick ({t1_actual_top}): {t1_actual_top}")
        print(f"     -> Payoff: {actual_vs_actual[1]:+.4f} (T1 win: {actual_vs_actual[3]:.1%})")
        
        if t1_best_response_to_actual == t1_actual_top:
            print(f"   [OK] Red picked the best response!")
        else:
            payoff_diff = actual_vs_actual[1] - actual_vs_br_actual[1]
            win_diff = actual_vs_actual[3] - actual_vs_br_actual[3]
            print(f"   [INFO] Red's actual pick differs from best response")
            if payoff_diff > 0:
                print(f"   Actual pick has payoff {payoff_diff:+.4f} MORE POSITIVE (worse for Red)")
            else:
                print(f"   Actual pick has payoff {abs(payoff_diff):+.4f} MORE NEGATIVE (better for Red)")
            print(f"   Win rate difference: {win_diff:+.1%}")
        
        print(f"\n" + "=" * 70)
        print("SPNE VERIFICATION")
        print("=" * 70)
        print(f"\nSPNE Definition:")
        print(f"  In a Subgame Perfect Nash Equilibrium, each player's strategy must be")
        print(f"  a best response to the other player's strategy at EVERY subgame.")
        print(f"  If Blue picks optimally ({optimal_kt_bot}, {optimal_kt_sup}), then")
        print(f"  Red's SPNE pick ({optimal_t1_top}) MUST be the best response (minimum payoff).")
        
        if t1_best_response_to_optimal == optimal_t1_top:
            print(f"\n[OK] SPNE VERIFIED: The SPNE pick ({optimal_t1_top}) IS the best response ({t1_best_response_to_optimal})!")
            print(f"  Payoff: {opt_vs_spne[1]:+.4f} (minimum = best for Red)")
        else:
            payoff_diff = opt_vs_br_opt[1] - opt_vs_spne[1]
            print(f"\n[WARNING] SPNE DISCREPANCY DETECTED!")
            print(f"  SPNE calculated ({optimal_t1_top}): {optimal_t1_top}")
            print(f"    -> Payoff: {opt_vs_spne[1]:+.4f} (T1 win: {opt_vs_spne[3]:.1%})")
            print(f"  Actual best response ({t1_best_response_to_optimal}): {t1_best_response_to_optimal}")
            print(f"    -> Payoff: {opt_vs_br_opt[1]:+.4f} (T1 win: {opt_vs_br_opt[3]:.1%})")
            print(f"  Payoff difference: {payoff_diff:+.4f} ({'more negative' if payoff_diff < 0 else 'less negative'} = better for Red)")
            print(f"  Win rate difference: {opt_vs_br_opt[3] - opt_vs_spne[3]:+.1%}")
            print(f"\n  This suggests the SPNE solver may have:")
            print(f"    1. Not explored all options (beam_width may need to be higher)")
            print(f"    2. Used inaccurate heuristics during candidate ranking")
            print(f"    3. Been affected by caching (though caching is now disabled)")
            print(f"    4. Red should pick the champion with the MOST NEGATIVE payoff")
        
        return result
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        scorer.close()


if __name__ == "__main__":
    result = run_nash_equilibrium_analysis()

