"""
Subgame Perfect Nash Equilibrium (SPNE) Solver for League of Legends
Uses backward induction to find optimal strategies in sequential games.

SPNE is a refinement of Nash equilibrium that eliminates non-credible threats
by requiring equilibrium strategies to be optimal at every subgame.
"""

from typing import List, Dict, Tuple, Optional, Set, Callable
from composition_scorer import CompositionScorer
import math
import hashlib
from copy import deepcopy


class SubgamePerfectNashEquilibrium:
    """
    Solver for Subgame Perfect Nash Equilibrium using backward induction.
    
    SPNE is solved by:
    1. Starting from the final decision nodes (leaves of game tree)
    2. Working backwards to find optimal strategies at each stage
    3. Each player chooses best response given optimal future play
    """
    
    def __init__(self, scorer: CompositionScorer, 
                 w1: float = 0.35, w2: float = 0.35, w3: float = 0.3,
                 use_memoization: bool = True,
                 skip_api_calls: bool = False,
                 show_progress: bool = True,
                 beam_width: int = 5,
                 fast_heuristic: bool = True):
        """
        Initialize SPNE solver.
        
        Args:
            scorer: CompositionScorer instance for calculating payoffs
            w1: Weight for lane matchups
            w2: Weight for player comfort
            w3: Weight for composition score
            use_memoization: If True, cache payoff calculations (default True)
            skip_api_calls: If True, skip slow API calls and use defaults (default False)
            show_progress: If True, show progress during calculation (default True)
            beam_width: Number of top candidates to explore at each level (default 5)
                      Reduces search space from O(n^d) to O(beam_width^d)
            fast_heuristic: If True, use simplified heuristics for candidate ranking (default True)
                          False uses full composition calculations (slower but more accurate)
        """
        self.scorer = scorer
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.use_memoization = use_memoization
        self.skip_api_calls = skip_api_calls
        self.show_progress = show_progress
        self.beam_width = beam_width
        self.payoff_cache = {}  # Cache for payoff calculations
        self.state_cache = {}  # Cache for game state evaluations
        self.calculation_count = 0  # Track number of calculations
        self.champion_data_cache = {}  # Cache for champion data lookups
        self.fast_heuristic = fast_heuristic  # Use simplified heuristics (faster but less accurate)
    
    def _get_champion_data_cached(self, champion_name: str) -> Optional[Dict]:
        """Get champion data with caching to avoid repeated database lookups."""
        if champion_name not in self.champion_data_cache:
            self.champion_data_cache[champion_name] = self.scorer.get_champion_data(champion_name)
        return self.champion_data_cache[champion_name]
    
    def calculate_payoff(self, blue_team: List[str], red_team: List[str],
                         blue_players: List[str] = None, 
                         red_players: List[str] = None,
                         region: str = 'na1',
                         use_fast_mode: bool = False) -> float:
        """
        Calculate payoff for blue team (red team payoff is negative of this).
        
        Payoff = S_score = w1(LaneMatchups) + w2(Comfort) + w3(TeamComp)
        This represents blue team's advantage.
        
        Args:
            blue_team: List of 5 blue side champions
            red_team: List of 5 red side champions
            blue_players: Optional list of blue player names
            red_players: Optional list of red player names
            region: Region for data fetching
            use_fast_mode: If True, skip expensive calculations (matchup/comfort)
            
        Returns:
            Payoff value (positive = blue advantage, negative = red advantage)
        """
        if len(blue_team) != 5 or len(red_team) != 5:
            return 0.0
        
        # Check cache if memoization is enabled
        if self.use_memoization:
            cache_key = (tuple(sorted(blue_team)), tuple(sorted(red_team)), 
                        tuple(blue_players) if blue_players else None,
                        tuple(red_players) if red_players else None,
                        region)
            if cache_key in self.payoff_cache:
                return self.payoff_cache[cache_key]
        
        # Track calculations
        self.calculation_count += 1
        if self.show_progress and self.calculation_count % 25 == 0:
            print(f"  Calculations: {self.calculation_count}...", end='\r', flush=True)
        
        # Calculate lane matchup score
        # If skip_api_calls or use_fast_mode, we'll use a simplified version
        if self.skip_api_calls or use_fast_mode:
            # Use composition score only (fast, no API calls)
            matchup = 0.0  # Skip slow matchup calculations
            comfort = 0.0  # Skip player comfort (requires API)
        else:
            matchup = self.scorer.calculate_lane_matchup_score(
                blue_team, red_team, None, self.w1, return_details=False
            )
            
            # Calculate player comfort score (if players provided)
            comfort = 0.0
            if blue_players and red_players and len(blue_players) == 5 and len(red_players) == 5:
                comfort_result = self.scorer.calculate_player_comfort_score(
                    blue_players, red_players, blue_team, red_team,
                    region, self.w2, return_details=False
                )
                comfort = comfort_result if isinstance(comfort_result, (int, float)) else comfort_result.get('score', 0.0)
        
        # Calculate composition scores (fast, uses database only)
        blue_comp = self.scorer.calculate_composition_score(blue_team, self.w3)
        red_comp = self.scorer.calculate_composition_score(red_team, self.w3)
        comp_advantage = blue_comp['total_score'] - red_comp['total_score']
        
        # Total payoff = S score
        payoff = matchup + comfort + comp_advantage
        
        # Cache result if memoization is enabled
        if self.use_memoization:
            self.payoff_cache[cache_key] = payoff
        
        return payoff
    
    def find_spne_sequential_draft(self, 
                                   available_champions: List[str],
                                   draft_order: List[str],
                                   blue_team: List[str] = None,
                                   red_team: List[str] = None,
                                   blue_players: List[str] = None,
                                   red_players: List[str] = None,
                                   region: str = 'na1',
                                   max_depth: int = None,
                                   max_picks: int = None,
                                   auto_ban_counters: bool = True,
                                   bans_per_pick: int = 1) -> Dict:
        """
        Find SPNE for a sequential draft game using backward induction.
        
        Args:
            available_champions: List of champions that can be picked
            draft_order: List of 'blue' or 'red' indicating whose turn it is
            blue_team: Current blue team (partial or complete)
            red_team: Current red team (partial or complete)
            blue_players: Optional blue player names
            red_players: Optional red player names
            region: Region for data fetching
            max_depth: Maximum depth to search from start (None = full tree)
                      Note: This limits how many picks are explored via backward induction.
                      Remaining picks are filled using a greedy heuristic.
            max_picks: Alternative to max_depth - limit number of picks to explore
                      (converted to max_depth based on current state)
            auto_ban_counters: If True, automatically ban champions with high winrate vs picked champions
            bans_per_pick: Number of bans to make per pick (default 1)
            
        Returns:
            Dictionary with SPNE strategies and payoffs
        """
        if blue_team is None:
            blue_team = []
        if red_team is None:
            red_team = []
        
        # Complete teams if needed
        while len(blue_team) < 5:
            blue_team.append(None)
        while len(red_team) < 5:
            red_team.append(None)
        
        # Filter out None values for current state
        blue_current = [c for c in blue_team if c is not None]
        red_current = [c for c in red_team if c is not None]
        
        # Find remaining picks needed
        picks_remaining = 5 - len(blue_current) + 5 - len(red_current)
        
        # If max_picks is specified, convert it to max_depth
        if max_picks is not None:
            max_depth = max_picks
        elif max_depth is None:
            max_depth = picks_remaining
        
        # Track bans (champions that should be excluded due to counter bans)
        banned_champions = set()
        
        # Get available picks (not already selected)
        selected = set(blue_current + red_current)
        available = [c for c in available_champions if c not in selected]
        
        # Auto-ban counters if enabled
        if auto_ban_counters:
            # Ban counters for already picked champions
            for champ in blue_current:
                counters = self._get_counter_champions(champ, available, bans_per_pick)
                banned_champions.update(counters)
            for champ in red_current:
                counters = self._get_counter_champions(champ, available, bans_per_pick)
                banned_champions.update(counters)
            
            # Remove banned champions from available
            available = [c for c in available if c not in banned_champions]
            
            if self.show_progress and banned_champions:
                print(f"  Auto-banned {len(banned_champions)} counter champions")
        
        # Clear caches if memoization is enabled
        if self.use_memoization:
            self.payoff_cache.clear()
            self.state_cache.clear()
        
        # Clear champion data cache (will be repopulated as needed)
        self.champion_data_cache.clear()
        
        # Reset calculation counter
        self.calculation_count = 0
        
        if self.show_progress:
            picks_explored = min(max_depth, picks_remaining)
            picks_heuristic = max(0, picks_remaining - max_depth)
            print(f"Starting backward induction:")
            print(f"  - Will explore {picks_explored} picks via backward induction")
            if picks_heuristic > 0:
                print(f"  - Will fill {picks_heuristic} picks using greedy heuristic")
            print(f"  - {len(available)} champions available")
        
        # Solve using backward induction
        result = self._backward_induction(
            available, draft_order, blue_team, red_team,
            blue_players, red_players, region, max_depth, 0,
            auto_ban_counters=auto_ban_counters,
            bans_per_pick=bans_per_pick,
            banned_champions=banned_champions.copy()
        )
        
        if self.show_progress:
            print()  # New line after progress updates
            print(f"  Completed: {self.calculation_count} payoff calculations")
        
        # Add cache statistics to result
        if self.use_memoization:
            result['cache_stats'] = {
                'payoff_cache_size': len(self.payoff_cache),
                'state_cache_size': len(self.state_cache),
                'total_calculations': self.calculation_count,
                'cache_hit_rate': f"{(len(self.payoff_cache) / max(1, self.calculation_count)) * 100:.1f}%"
            }
        
        # Add banned champions to result if auto-banning was enabled
        if auto_ban_counters:
            result['banned_champions'] = list(banned_champions)
            result['auto_bans_enabled'] = True
        else:
            result['banned_champions'] = []
            result['auto_bans_enabled'] = False
        
        return result
    
    def _get_counter_champions(self, picked_champion: str, available_champions: List[str], 
                               num_bans: int = 1, winrate_threshold: float = 0.52) -> List[str]:
        """
        Find champions that counter the picked champion (have high winrate vs it).
        
        Args:
            picked_champion: The champion that was just picked
            available_champions: List of available champions to check
            num_bans: Number of counter champions to ban
            winrate_threshold: Minimum winrate to consider a counter (default 0.52 = 52%)
            
        Returns:
            List of champion names that counter the picked champion
        """
        role_slots = ['top', 'jungle', 'mid', 'adc', 'support']
        role_mapping = {
            'top': 'is_top',
            'jungle': 'is_jungle',
            'mid': 'is_mid',
            'adc': 'is_adc',
            'support': 'is_support'
        }
        
        # Determine which role the picked champion plays
        picked_role = None
        picked_data = self._get_champion_data_cached(picked_champion)
        if picked_data:
            for role_name, role_key in role_mapping.items():
                if picked_data.get(role_key, 0) == 1:
                    picked_role = role_name
                    break
        
        # Find champions with high winrate vs the picked champion
        counter_scores = []
        for opponent in available_champions:
            # Skip if opponent is the same as picked champion
            if opponent == picked_champion:
                continue
            
            # Get matchup winrate (opponent's winrate vs picked champion)
            # If picked_role is None, try all roles and take the worst matchup
            worst_winrate = None
            for role in ([picked_role] if picked_role else role_slots):
                try:
                    # Get opponent's winrate vs picked champion
                    wr = self.scorer.get_champion_matchup_winrate(opponent, picked_champion, role)
                    if wr is not None:
                        if worst_winrate is None or wr > worst_winrate:
                            worst_winrate = wr
                except:
                    continue
            
            # If we found a matchup winrate above threshold, it's a counter
            if worst_winrate is not None and worst_winrate >= winrate_threshold:
                counter_scores.append((worst_winrate, opponent))
        
        # Sort by winrate (highest first) and return top num_bans
        counter_scores.sort(reverse=True, key=lambda x: x[0])
        return [champ for _, champ in counter_scores[:num_bans]]
    
    def _backward_induction(self,
                           available_champions: List[str],
                           draft_order: List[str],
                           blue_team: List[str],
                           red_team: List[str],
                           blue_players: List[str],
                           red_players: List[str],
                           region: str,
                           max_depth: int,
                           current_depth: int,
                           alpha: float = float('-inf'),
                           beta: float = float('inf'),
                           auto_ban_counters: bool = True,
                           bans_per_pick: int = 1,
                           banned_champions: Set[str] = None) -> Dict:
        """
        Recursive backward induction algorithm with state caching and alpha-beta pruning.
        
        Returns optimal strategy and payoff from current game state.
        """
        if banned_champions is None:
            banned_champions = set()
        # Base case: if teams are complete or max depth reached
        blue_current = [c for c in blue_team if c is not None]
        red_current = [c for c in red_team if c is not None]
        
        # Determine whose turn it is (needed for cache key)
        blue_picks = len(blue_current)
        red_picks = len(red_current)
        total_picks = blue_picks + red_picks
        
        # Show progress for current depth
        # Note: The picks count can go up and down as we explore different branches - this is normal for backtracking
        # We're exploring the game tree recursively, so we'll see different states as we go deeper and backtrack
        # Even though we see "3 picks" multiple times, each visit represents a DIFFERENT combination of champions
        # (e.g., [A,B,C] vs [A,D,C] vs [E,B,C] - all are 3 picks but different game states)
        if self.show_progress and current_depth < max_depth and total_picks < 10:
            # Show progress less frequently to avoid confusion from backtracking
            if self.calculation_count % 25 == 0:
                blue_str = ','.join(blue_current[:2]) if len(blue_current) > 0 else 'none'
                red_str = ','.join(red_current[:2]) if len(red_current) > 0 else 'none'
                print(f"  Depth {current_depth}/{max_depth} | {total_picks} picks (B:{blue_str} R:{red_str}) | Calc: {self.calculation_count}...", end='\r', flush=True)
        
        # Create state cache key
        # Note: We don't include available_champions in the key because:
        # 1. It makes cache less effective (same state with different available champs won't match)
        # 2. Available champs can be derived from blue/red teams and initial pool
        # 3. The optimal strategy at a given state is primarily determined by team composition
        state_key = (
            tuple(sorted(blue_current)), tuple(sorted(red_current)),
            total_picks,  # Track position in draft order
            tuple(blue_players) if blue_players else None,
            tuple(red_players) if red_players else None,
            region
        )
        
        # Check state cache if memoization is enabled
        if self.use_memoization and state_key in self.state_cache:
            cached_result = self.state_cache[state_key]
            return cached_result.copy()
        
        if len(blue_current) == 5 and len(red_current) == 5:
            # Game complete - calculate payoff (use full calculation for terminal nodes)
            payoff = self.calculate_payoff(blue_current, red_current,
                                          blue_players, red_players, region, use_fast_mode=False)
            result = {
                'blue_team': blue_current,
                'red_team': red_current,
                'payoff': payoff,
                'strategy': None,
                'best_response': None
            }
            if self.use_memoization:
                self.state_cache[state_key] = result.copy()
            return result
        
        if current_depth >= max_depth or not draft_order:
            # Reached max depth - evaluate current state using fast mode
            # Fill remaining slots with best available or None
            # Filter out banned champions
            available_filtered = [c for c in available_champions if c not in banned_champions]
            payoff, completed_blue, completed_red = self._evaluate_partial_state(
                blue_team, red_team, available_filtered,
                blue_players, red_players, region, use_fast_mode=True
            )
            result = {
                'blue_team': completed_blue,
                'red_team': completed_red,
                'payoff': payoff,
                'strategy': None,
                'best_response': None
            }
            if self.use_memoization:
                self.state_cache[state_key] = result.copy()
            return result
        
        if total_picks >= len(draft_order):
            # Draft order exhausted, fill remaining using fast mode
            # Filter out banned champions
            available_filtered = [c for c in available_champions if c not in banned_champions]
            payoff, completed_blue, completed_red = self._evaluate_partial_state(
                blue_team, red_team, available_filtered,
                blue_players, red_players, region, use_fast_mode=True
            )
            result = {
                'blue_team': completed_blue,
                'red_team': completed_red,
                'payoff': payoff,
                'strategy': None,
                'best_response': None
            }
            if self.use_memoization:
                self.state_cache[state_key] = result.copy()
            return result
        
        current_player = draft_order[total_picks]
        is_blue = (current_player == 'blue')
        
        # Use beam search: evaluate candidates and only explore top N
        candidates_to_explore = available_champions
        if len(available_champions) > self.beam_width:
            # Determine which role slot needs to be filled
            role_slots = ['top', 'jungle', 'mid', 'adc', 'support']
            if is_blue:
                empty_slot_idx = next((i for i, c in enumerate(blue_team) if c is None), None)
            else:
                empty_slot_idx = next((i for i, c in enumerate(red_team) if c is None), None)
            
            target_role = role_slots[empty_slot_idx] if empty_slot_idx is not None else None
            
            # Map role names to database column names
            role_mapping = {
                'top': 'is_top',
                'jungle': 'is_jungle',
                'mid': 'is_mid',
                'adc': 'is_adc',
                'support': 'is_support'
            }
            role_key = role_mapping.get(target_role) if target_role else None
            
            # FIRST: Filter to only champions that can fill the required role
            valid_role_champions = []
            invalid_role_champions = []
            
            for champ in available_champions:
                champ_data = self.scorer.get_champion_data(champ)
                if champ_data is None:
                    # If we can't get data, treat as invalid for specific roles
                    if role_key:
                        invalid_role_champions.append(champ)
                    else:
                        valid_role_champions.append(champ)
                    continue
                
                # If we have a target role, check if champion can fill it
                if role_key:
                    # Check role flag (1 = can play this role)
                    role_value = champ_data.get(role_key)
                    
                    # Check if role data exists for this champion
                    if role_key not in champ_data or role_value is None:
                        # Role data missing - check if champion has ANY role data
                        has_any_role_data = any(
                            champ_data.get(f'is_{r}', 0) == 1 
                            for r in ['top', 'jungle', 'mid', 'adc', 'support']
                        )
                        if has_any_role_data:
                            # Champion has role data but not for this role - invalid
                            can_fill_role = False
                        else:
                            # No role data at all - be lenient and allow it (data might be incomplete)
                            can_fill_role = True
                    else:
                        # Role data exists - check if it's 1 (can play this role)
                        can_fill_role = role_value == 1 or role_value == '1' or str(role_value).strip() == '1'
                    
                    if can_fill_role:
                        valid_role_champions.append(champ)
                    else:
                        invalid_role_champions.append(champ)
                else:
                    # No specific role requirement, all are valid
                    valid_role_champions.append(champ)
            
            # Prioritize valid role champions, but include invalid ones if needed
            candidates_to_rank = valid_role_champions
            if len(candidates_to_rank) < self.beam_width:
                # Not enough valid role champions, add some invalid ones as fallback
                candidates_to_rank = valid_role_champions + invalid_role_champions[:self.beam_width - len(valid_role_champions)]
            
            # Quick evaluation to rank candidates - simplified for speed
            candidate_scores = []
            for champ in candidates_to_rank:
                # Get champion role data (cached)
                champ_data = self._get_champion_data_cached(champ)
                if champ_data is None:
                    candidate_scores.append((float('-inf') if is_blue else float('inf'), champ))
                    continue
                
                # Check if champion can fill the target role (for scoring)
                can_fill_role = False
                if role_key:
                    can_fill_role = champ_data.get(role_key, 0) == 1
                
                # Check if this role is missing from the team (cached lookups)
                team_to_check = blue_team if is_blue else red_team
                team_current = [c for c in team_to_check if c is not None]
                missing_role_bonus = 0.0
                if target_role and len(team_current) < 5 and can_fill_role:
                    # Check if this role is already filled (using cached data)
                    role_filled = False
                    for existing_champ in team_current:
                        existing_data = self._get_champion_data_cached(existing_champ)
                        if existing_data and existing_data.get(role_key, 0) == 1:
                            role_filled = True
                            break
                    
                    if not role_filled:
                        missing_role_bonus = 500.0  # Bonus for filling missing role
                
                # Simplified heuristic: use lightweight scoring instead of full composition
                if self.fast_heuristic:
                    # Fast mode: use simple attribute-based scoring
                    champ_value = 0.0
                    if champ_data.get('is_hypercarry', 0) == 1:
                        champ_value += 10.0
                    if champ_data.get('is_scaling', 0) == 1:
                        champ_value += 5.0
                    if champ_data.get('has_peel', 0) == 1:
                        champ_value += 5.0
                    if champ_data.get('has_aoe_combo', 0) == 1:
                        champ_value += 5.0
                    if champ_data.get('has_wave_clear', 0) == 1:
                        champ_value += 3.0
                    
                    composition_score = champ_value if is_blue else -champ_value
                else:
                    # Full mode: calculate actual composition (slower but more accurate)
                    composition_score = 0.0
                    try:
                        test_blue = blue_team.copy()
                        test_red = red_team.copy()
                        if is_blue and empty_slot_idx is not None:
                            test_blue[empty_slot_idx] = champ
                        elif not is_blue and empty_slot_idx is not None:
                            test_red[empty_slot_idx] = champ
                        
                        test_blue_curr = [c for c in test_blue if c is not None]
                        test_red_curr = [c for c in test_red if c is not None]
                        
                        # Only calculate composition if we have at least some champions
                        if len(test_blue_curr) > 0 or len(test_red_curr) > 0:
                            # For complete teams, use full composition score
                            if len(test_blue_curr) == 5 and len(test_red_curr) == 5:
                                blue_comp = self.scorer.calculate_composition_score(test_blue_curr, self.w3)
                                red_comp = self.scorer.calculate_composition_score(test_red_curr, self.w3)
                                composition_score = blue_comp['total_score'] - red_comp['total_score']
                            else:
                                # Partial team: use weighted composition scores
                                if len(test_blue_curr) > 0:
                                    try:
                                        blue_comp = self.scorer.calculate_composition_score(test_blue_curr, self.w3)
                                        composition_score += blue_comp.get('total_score', 0) * (len(test_blue_curr) / 5.0)
                                    except:
                                        pass
                                if len(test_red_curr) > 0:
                                    try:
                                        red_comp = self.scorer.calculate_composition_score(test_red_curr, self.w3)
                                        composition_score -= red_comp.get('total_score', 0) * (len(test_red_curr) / 5.0)
                                    except:
                                        pass
                    except:
                        composition_score = 0.0
                
                # Total heuristic score: role fit bonus + missing role bonus + composition
                # Valid role champions get a large bonus to ensure they're prioritized
                role_fit_bonus = 1000.0 if can_fill_role else 0.0
                heuristic_score = role_fit_bonus + missing_role_bonus + composition_score
                if not is_blue:
                    heuristic_score = -heuristic_score  # Red minimizes
                
                # Add small tie-breaker to avoid alphabetical bias when scores are equal
                # Use hash of champion name for deterministic but non-alphabetical ordering
                tie_breaker = int(hashlib.md5(champ.encode()).hexdigest()[:8], 16) / 1e10
                heuristic_score += tie_breaker
                
                candidate_scores.append((heuristic_score, champ))
            
            # Sort by score and take top beam_width
            # Use secondary sort by champion name (reversed) to break ties non-alphabetically
            candidate_scores.sort(key=lambda x: (x[0], x[1][::-1]), reverse=is_blue)
            candidates_to_explore = [champ for _, champ in candidate_scores[:self.beam_width]]
        
        # Find best response by trying top candidates
        best_payoff = float('-inf') if is_blue else float('inf')
        best_action = None
        best_future = None
        
        # Determine which role slot needs to be filled (for consistent placement)
        if is_blue:
            empty_slot_idx = next((i for i, c in enumerate(blue_team) if c is None), None)
        else:
            empty_slot_idx = next((i for i, c in enumerate(red_team) if c is None), None)
        
        # Try picking each candidate champion
        for champ in candidates_to_explore:
            # Validate role assignment: champion must be able to play the target role
            if empty_slot_idx is not None:
                role_slots = ['top', 'jungle', 'mid', 'adc', 'support']
                target_role = role_slots[empty_slot_idx]
                role_mapping = {
                    'top': 'is_top',
                    'jungle': 'is_jungle',
                    'mid': 'is_mid',
                    'adc': 'is_adc',
                    'support': 'is_support'
                }
                role_key = role_mapping.get(target_role)
                
                # Check if champion can play this role
                champ_data = self._get_champion_data_cached(champ)
                if champ_data and role_key:
                    role_value = champ_data.get(role_key, 0)
                    can_play_role = (role_value == 1 or role_value == '1' or str(role_value).strip() == '1')
                    
                    # Skip if champion cannot play the required role
                    if not can_play_role:
                        continue
            
            # Create new state
            new_blue = blue_team.copy()
            new_red = red_team.copy()
            new_available = [c for c in available_champions if c != champ]
            new_banned = banned_champions.copy()
            
            # Auto-ban counters for the newly picked champion
            if auto_ban_counters:
                counters = self._get_counter_champions(champ, new_available, bans_per_pick)
                new_banned.update(counters)
                new_available = [c for c in new_available if c not in new_banned]
            
            # Add champion to appropriate team in the correct role slot
            if is_blue and empty_slot_idx is not None:
                new_blue[empty_slot_idx] = champ
            elif not is_blue and empty_slot_idx is not None:
                new_red[empty_slot_idx] = champ
            else:
                # Fallback: find first empty slot (shouldn't happen if role constraints are working)
                if is_blue:
                    for i in range(5):
                        if new_blue[i] is None:
                            new_blue[i] = champ
                            break
                else:
                    for i in range(5):
                        if new_red[i] is None:
                            new_red[i] = champ
                            break
            
            # Recursively solve subgame
            subgame_result = self._backward_induction(
                new_available, draft_order, new_blue, new_red,
                blue_players, red_players, region, max_depth, current_depth + 1,
                alpha, beta, auto_ban_counters, bans_per_pick, new_banned
            )
            
            subgame_payoff = subgame_result['payoff']
            
            # Alpha-beta pruning
            if is_blue:
                if subgame_payoff > best_payoff:
                    best_payoff = subgame_payoff
                    best_action = champ
                    best_future = subgame_result
                    alpha = max(alpha, best_payoff)
                # Prune if Red can guarantee a better outcome elsewhere
                if best_payoff >= beta:
                    break
            else:
                if subgame_payoff < best_payoff:
                    best_payoff = subgame_payoff
                    best_action = champ
                    best_future = subgame_result
                    beta = min(beta, best_payoff)
                # Prune if Blue can guarantee a better outcome elsewhere
                if best_payoff <= alpha:
                    break
        
        # Return best response
        if is_blue:
            final_blue = best_future['blue_team'] if best_future else blue_current
            final_red = best_future['red_team'] if best_future else red_current
        else:
            final_blue = best_future['blue_team'] if best_future else blue_current
            final_red = best_future['red_team'] if best_future else red_current
        
        result = {
            'blue_team': final_blue,
            'red_team': final_red,
            'payoff': best_payoff,
            'strategy': best_action,
            'best_response': best_future,
            'current_player': current_player,
            'optimal_action': best_action
        }
        
        # Cache result
        if self.use_memoization:
            self.state_cache[state_key] = result.copy()
        
        return result
    
    def _evaluate_partial_state(self,
                                blue_team: List[str],
                                red_team: List[str],
                                available_champions: List[str],
                                blue_players: List[str],
                                red_players: List[str],
                                region: str,
                                use_fast_mode: bool = False) -> Tuple[float, List[str], List[str]]:
        """
        Evaluate a partial game state by filling remaining slots optimally.
        Uses greedy heuristic: fill with best available champions.
        
        Returns:
            Tuple of (payoff, completed_blue_team, completed_red_team)
        """
        blue_current = [c for c in blue_team if c is not None]
        red_current = [c for c in red_team if c is not None]
        
        # If teams are complete, calculate payoff
        if len(blue_current) == 5 and len(red_current) == 5:
            payoff = self.calculate_payoff(blue_current, red_current,
                                        blue_players, red_players, region, use_fast_mode=use_fast_mode)
            return payoff, blue_current, red_current
        
        # Greedy completion: fill remaining slots with role-appropriate champions
        # Reconstruct full team arrays to track which slots are filled
        blue_complete = blue_team.copy()
        red_complete = red_team.copy()
        remaining = available_champions.copy()
        
        role_slots = ['top', 'jungle', 'mid', 'adc', 'support']
        role_mapping = {
            'top': 'is_top',
            'jungle': 'is_jungle',
            'mid': 'is_mid',
            'adc': 'is_adc',
            'support': 'is_support'
        }
        
        # Fill each team with role-appropriate champions
        for team_complete, is_blue_team in [(blue_complete, True), (red_complete, False)]:
            # Fill each empty slot with a champion that can play that role
            for slot_idx in range(5):
                if team_complete[slot_idx] is None:
                    target_role = role_slots[slot_idx]
                    role_key = role_mapping[target_role]
                    
                    # Find champions that can play this role
                    valid_champions = []
                    for champ in remaining:
                        champ_data = self._get_champion_data_cached(champ)
                        if champ_data:
                            role_value = champ_data.get(role_key, 0)
                            can_play = (role_value == 1 or role_value == '1' or str(role_value).strip() == '1')
                            if can_play:
                                valid_champions.append(champ)
                    
                    # If we have valid champions, pick one; otherwise use any available
                    if valid_champions:
                        # In fast mode, just pick first valid; otherwise evaluate
                        if use_fast_mode:
                            selected = valid_champions[0]
                        else:
                            # Evaluate which valid champion gives best composition
                            best_champ = None
                            best_score = float('-inf')
                            for champ in valid_champions:
                                test_team = team_complete.copy()
                                test_team[slot_idx] = champ
                                test_team_curr = [c for c in test_team if c is not None]
                                if len(test_team_curr) < 5:
                                    test_team_curr = test_team_curr + ["Unknown"] * (5 - len(test_team_curr))
                                try:
                                    comp_score = self.scorer.calculate_composition_score(test_team_curr, self.w3)
                                    if comp_score['total_score'] > best_score:
                                        best_score = comp_score['total_score']
                                        best_champ = champ
                                except:
                                    if best_champ is None:
                                        best_champ = champ
                            selected = best_champ if best_champ else valid_champions[0]
                    else:
                        # No valid champions for this role - use any available (will be penalized)
                        selected = remaining[0] if remaining else None
                    
                    if selected:
                        team_complete[slot_idx] = selected
                        remaining.remove(selected)
        
        # Convert back to list format (remove None values)
        blue_final = [c for c in blue_complete if c is not None]
        red_final = [c for c in red_complete if c is not None]
        
        # If still incomplete, use placeholders (will result in lower payoff)
        while len(blue_final) < 5:
            blue_final.append("Unknown")
        while len(red_final) < 5:
            red_final.append("Unknown")
        
        try:
            payoff = self.calculate_payoff(blue_final, red_final,
                                        blue_players, red_players, region, use_fast_mode=use_fast_mode)
            return payoff, blue_final, red_final
        except:
            # If calculation fails, return neutral payoff
            return 0.0, blue_final, red_final
    
    def find_spne_simultaneous(self,
                               available_champions: List[str],
                               blue_team: List[str] = None,
                               red_team: List[str] = None,
                               blue_players: List[str] = None,
                               red_players: List[str] = None,
                               region: str = 'na1',
                               role_constraints: Dict[str, List[str]] = None) -> Dict:
        """
        Find Nash equilibrium for simultaneous game (all picks at once).
        Uses best response dynamics to converge to equilibrium.
        
        Args:
            available_champions: List of champions that can be picked
            blue_team: Initial blue team (can be partial)
            red_team: Initial red team (can be partial)
            blue_players: Optional blue player names
            red_players: Optional red player names
            region: Region for data fetching
            role_constraints: Dict mapping role to list of valid champions
            
        Returns:
            Dictionary with Nash equilibrium strategies
        """
        if blue_team is None:
            blue_team = []
        if red_team is None:
            red_team = []
        
        # Complete teams randomly if needed
        while len(blue_team) < 5:
            blue_team.append(None)
        while len(red_team) < 5:
            red_team.append(None)
        
        # Get available champions
        selected = set([c for c in blue_team + red_team if c is not None])
        available = [c for c in available_champions if c not in selected]
        
        # Best response dynamics
        max_iterations = 100
        convergence_threshold = 0.001
        
        for iteration in range(max_iterations):
            # Blue team best response
            blue_best = self._best_response(
                red_team, blue_team, available, True,
                blue_players, red_players, region, role_constraints
            )
            
            # Red team best response
            red_best = self._best_response(
                blue_team, red_team, available, False,
                blue_players, red_players, region, role_constraints
            )
            
            # Check convergence
            blue_changed = blue_best != blue_team
            red_changed = red_best != red_team
            
            if not blue_changed and not red_changed:
                # Converged to equilibrium
                blue_final = [c for c in blue_best if c is not None]
                red_final = [c for c in red_best if c is not None]
                payoff = self.calculate_payoff(blue_final, red_final,
                                              blue_players, red_players, region)
                
                return {
                    'blue_team': blue_final,
                    'red_team': red_final,
                    'payoff': payoff,
                    'iterations': iteration + 1,
                    'converged': True
                }
            
            blue_team = blue_best
            red_team = red_best
        
        # Did not converge
        blue_final = [c for c in blue_team if c is not None]
        red_final = [c for c in red_team if c is not None]
        payoff = self.calculate_payoff(blue_final, red_final,
                                      blue_players, red_players, region)
        
        return {
            'blue_team': blue_final,
            'red_team': red_final,
            'payoff': payoff,
            'iterations': max_iterations,
            'converged': False
        }
    
    def _best_response(self,
                      opponent_team: List[str],
                      my_team: List[str],
                      available: List[str],
                      is_blue: bool,
                      blue_players: List[str],
                      red_players: List[str],
                      region: str,
                      role_constraints: Dict[str, List[str]] = None) -> List[str]:
        """
        Find best response strategy given opponent's team.
        """
        my_current = [c for c in my_team if c is not None]
        opponent_current = [c for c in opponent_team if c is not None]
        
        # If team is complete, return as is
        if len(my_current) == 5:
            return my_team.copy()
        
        # Try replacing each empty slot with best available champion
        best_team = my_team.copy()
        best_payoff = float('-inf') if is_blue else float('inf')
        
        # Find empty slots
        empty_indices = [i for i, c in enumerate(my_team) if c is None]
        
        # Try filling each empty slot
        for idx in empty_indices:
            for champ in available:
                if champ in my_team:
                    continue
                
                # Check role constraints if provided
                if role_constraints:
                    role = ['top', 'jungle', 'mid', 'adc', 'support'][idx]
                    if role in role_constraints:
                        if champ not in role_constraints[role]:
                            continue
                
                # Try this champion
                test_team = my_team.copy()
                test_team[idx] = champ
                test_current = [c for c in test_team if c is not None]
                
                # Calculate payoff
                if is_blue:
                    payoff = self.calculate_payoff(
                        test_current, opponent_current,
                        blue_players, red_players, region
                    )
                    if payoff > best_payoff:
                        best_payoff = payoff
                        best_team = test_team.copy()
                else:
                    payoff = self.calculate_payoff(
                        opponent_current, test_current,
                        blue_players, red_players, region
                    )
                    # Red minimizes blue's payoff
                    if payoff < best_payoff:
                        best_payoff = payoff
                        best_team = test_team.copy()
        
        return best_team
    
    def get_all_champions(self) -> List[str]:
        """Get list of all champions from database."""
        cursor = self.scorer.conn.cursor()
        cursor.execute("SELECT name FROM champions ORDER BY name")
        return [row[0] for row in cursor.fetchall()]


def example_usage():
    """Example of how to use SPNE solver."""
    scorer = CompositionScorer()
    spne = SubgamePerfectNashEquilibrium(scorer)
    
    # Get available champions
    all_champions = spne.get_all_champions()
    
    # Example: Sequential draft (standard League draft order)
    # Blue picks first, then Red picks 2, then Blue picks 2, etc.
    draft_order = ['blue', 'red', 'red', 'blue', 'blue', 
                   'red', 'red', 'blue', 'blue', 'red']
    
    # Find SPNE
    result = spne.find_spne_sequential_draft(
        available_champions=all_champions[:50],  # Limit for performance
        draft_order=draft_order,
        max_depth=10  # Limit depth for performance
    )
    
    print("SPNE Result:")
    print(f"Blue Team: {result['blue_team']}")
    print(f"Red Team: {result['red_team']}")
    print(f"Payoff (Blue advantage): {result['payoff']:.4f}")
    
    return result


if __name__ == "__main__":
    example_usage()

