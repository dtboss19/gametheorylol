"""
League of Legends Team Composition Scorer
Calculates team composition score based on various tags and synergies.
Score is constrained to [-0.15, 0.15] range.
"""

import sqlite3
import re
import requests
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup

# Import player comfort data from database
try:
    from get_worlds_data import get_player_comfort, load_player_comfort_data
    HAS_PLAYER_COMFORT_DB = True
except ImportError:
    HAS_PLAYER_COMFORT_DB = False
    def get_player_comfort(player_name, champion_name, min_games=5):
        return {'winrate': 0.5, 'games': 0, 'found': False}
from urllib.parse import urlparse, parse_qs


class CompositionScorer:
    # Global alpha constant for sigmoid function
    ALPHA = 0.02
    
    def __init__(self, db_path: str = "lolchampiontags.db", riot_api_key: str = None):
        """
        Initialize the scorer with database connection.
        
        Args:
            db_path: Path to the champion tags database
            riot_api_key: Optional Riot Games API key for fetching match data
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self.riot_api_key = riot_api_key
        # Cache for champion matchups: (champ1, champ2, role) -> winrate
        # This prevents redundant API calls for the same matchup
        self.matchup_cache = {}
        # Cache for player-champion winrates: (player_name, champion, region) -> {'winrate': float, 'games_played': int}
        # This prevents redundant OP.GG scraping for the same player-champion combination
        self.player_comfort_cache = {}
    
    def get_champion_data(self, champion_name: str) -> Optional[Dict]:
        """Get champion data from database by name."""
        cursor = self.conn.cursor()
        # Try to get role columns if they exist
        try:
            cursor.execute("""
                SELECT name, damage_type, engage_type, bruiser_position,
                       has_poke, has_aoe_combo, has_wave_clear,
                       is_team_fight, is_split_pusher, is_scaling,
                       has_peel, is_hypercarry, global_ability,
                       is_top, is_jungle, is_mid, is_adc, is_support, is_ranged
                FROM champions
                WHERE LOWER(name) = LOWER(?)
            """, (champion_name,))
        except sqlite3.OperationalError:
            # Role columns don't exist yet, use basic query
            cursor.execute("""
                SELECT name, damage_type, engage_type, bruiser_position,
                       has_poke, has_aoe_combo, has_wave_clear,
                       is_team_fight, is_split_pusher, is_scaling,
                       has_peel, is_hypercarry, global_ability
                FROM champions
                WHERE LOWER(name) = LOWER(?)
            """, (champion_name,))
        
        row = cursor.fetchone()
        if row:
            champ_dict = dict(row)
            # Infer roles if not present (fallback logic)
            if 'is_top' not in champ_dict or champ_dict.get('is_top') is None:
                champ_dict = self._infer_roles(champ_dict)
            return champ_dict
        return None
    
    def _infer_roles(self, champ: Dict) -> Dict:
        """Infer role flags from existing data if role columns don't exist."""
        # Default to 0 if not present
        champ.setdefault('is_top', 0)
        champ.setdefault('is_jungle', 0)
        champ.setdefault('is_mid', 0)
        champ.setdefault('is_adc', 0)
        champ.setdefault('is_support', 0)
        champ.setdefault('is_ranged', 0)
        
        # Infer some roles from tags (basic heuristics)
        # Support: has peel and not a carry
        if champ.get('has_peel', 0) == 1 and champ.get('is_hypercarry', 0) == 0:
            champ['is_support'] = 1
        
        # ADC: ranged, AD damage, hypercarry or scaling
        if (champ.get('damage_type', '').upper() == 'AD' and 
            champ.get('is_hypercarry', 0) == 1):
            champ['is_adc'] = 1
            champ['is_ranged'] = 1
        
        return champ
    
    def calculate_composition_score(self, champion_names: List[str], w3: float = 0.3) -> Dict:
        """
        Calculate the composition score for a team of champions.
        Score is normalized to [-0.15, 0.15] range, then multiplied by w3.
        
        Args:
            champion_names: List of 5 champion names
            w3: Weight factor for composition score (default 0.3)
            
        Returns:
            Dictionary with score breakdown and total score
        """
        if len(champion_names) != 5:
            raise ValueError("Team must have exactly 5 champions")
        
        # Get all champion data
        champions = []
        for name in champion_names:
            champ_data = self.get_champion_data(name)
            if champ_data is None:
                raise ValueError(f"Champion '{name}' not found in database")
            champions.append(champ_data)
        
        # Calculate counts and totals
        stats = self._calculate_stats(champions)
        
        # Calculate bonuses and penalties
        bonuses = {}
        penalties = {}
        
        # Role requirements (CRITICAL - highest priority)
        bonuses['role_requirements'], penalties['role_requirements'] = self._role_requirement_score(stats)
        
        # Core composition scores
        bonuses['wave_clear'], penalties['wave_clear'] = self._wave_clear_score(stats)
        bonuses['frontline'], penalties['frontline'] = self._frontline_score(stats)
        bonuses['damage_mix'], penalties['damage_mix'] = self._damage_mix_score(stats)
        bonuses['poke'], penalties['poke'] = self._poke_score(stats)
        bonuses['scaling'], penalties['scaling'] = self._scaling_score(stats)
        bonuses['pick'], penalties['pick'] = self._pick_score(stats)
        bonuses['aoe_combo'], penalties['aoe_combo'] = self._aoe_combo_score(stats)
        bonuses['team_fight'], penalties['team_fight'] = self._team_fight_score(stats)
        bonuses['split_pusher'], penalties['split_pusher'] = self._split_pusher_score(stats)
        bonuses['hypercarry_peel'], penalties['hypercarry_peel'] = self._hypercarry_peel_score(stats)
        bonuses['global'], penalties['global'] = self._global_score(stats)
        
        # New scoring categories
        bonuses['resource_allocation'], penalties['resource_allocation'] = self._resource_allocation_score(stats)
        bonuses['win_condition'], penalties['win_condition'] = self._win_condition_clarity_score(stats)
        
        # Calculate raw totals
        total_bonus = sum(bonuses.values())
        total_penalty = sum(penalties.values())
        raw_score = total_bonus - total_penalty
        
        # Normalize to [-0.15, 0.15] range
        normalized_score = self._normalize_score(raw_score)
        
        # Apply weight factor w3
        final_score = normalized_score * w3
        
        # Clamp to [-0.15, 0.15] range after applying w3
        final_score = max(-0.15, min(0.15, final_score))
        
        return {
            'champions': champion_names,
            'stats': stats,
            'bonuses': bonuses,
            'penalties': penalties,
            'total_bonus': total_bonus,
            'total_penalty': total_penalty,
            'raw_score': raw_score,
            'normalized_score': normalized_score,
            'w3': w3,
            'total_score': final_score
        }
    
    def _normalize_score(self, raw_score: float) -> float:
        """Normalize score to [-0.15, 0.15] range using sigmoid-like scaling."""
        # Use a scaling factor to map scores to the desired range
        # This ensures all scores fall within [-0.15, 0.15]
        # You can adjust the scaling factor based on typical score ranges
        
        # Simple clamping approach
        max_expected_range = 0.50  # Expected max deviation from 0
        scale_factor = 0.15 / max_expected_range
        
        normalized = raw_score * scale_factor
        
        # Hard clamp to ensure it's always in range
        return max(-0.15, min(0.15, normalized))
    
    def _calculate_stats(self, champions: List[Dict]) -> Dict:
        """Calculate aggregated statistics from champion list."""
        stats = {
            'wave_clear': 0,
            'frontline_count': 0,
            'tank_count': 0,
            'bruiser_front_count': 0,
            'bruiser_diver_count': 0,
            'bruiser_total': 0,
            'ad_count': 0,
            'ap_count': 0,
            'mixed_count': 0,
            'poke_count': 0,
            'scaling_count': 0,
            'pick_count': 0,
            'diver_count': 0,
            'aoe_combo_count': 0,
            'team_fight_count': 0,
            'split_pusher_count': 0,
            'peel_count': 0,
            'hypercarry_count': 0,
            'global_count': 0,
            # Role counts
            'top_count': 0,
            'jungle_count': 0,
            'mid_count': 0,
            'adc_count': 0,
            'support_count': 0,
            'ranged_adc_count': 0
        }
        
        for champ in champions:
            # Wave clear
            stats['wave_clear'] += champ.get('has_wave_clear', 0)
            
            # Frontline/Tank
            if champ.get('engage_type') == 'Tank':
                stats['tank_count'] += 1
                stats['frontline_count'] += 1
            
            # Bruiser positions
            if champ.get('bruiser_position') == 'Front':
                stats['bruiser_front_count'] += 1
                stats['bruiser_total'] += 1
                stats['frontline_count'] += 1
            elif champ.get('bruiser_position') == 'Diver':
                stats['bruiser_diver_count'] += 1
                stats['bruiser_total'] += 1
                stats['diver_count'] += 1
            
            # Damage type
            damage_type = champ.get('damage_type') or ''
            if damage_type.upper() == 'AD':
                stats['ad_count'] += 1
            elif damage_type.upper() == 'AP':
                stats['ap_count'] += 1
            elif damage_type.upper() == 'MIXED':
                stats['mixed_count'] += 1
            
            # Other tags
            stats['poke_count'] += champ.get('has_poke', 0)
            stats['scaling_count'] += champ.get('is_scaling', 0)
            stats['pick_count'] += (1 if champ.get('engage_type') == 'Pick' else 0)
            stats['aoe_combo_count'] += champ.get('has_aoe_combo', 0)
            stats['team_fight_count'] += champ.get('is_team_fight', 0)
            stats['split_pusher_count'] += champ.get('is_split_pusher', 0)
            stats['peel_count'] += champ.get('has_peel', 0)
            stats['hypercarry_count'] += champ.get('is_hypercarry', 0)
            stats['global_count'] += champ.get('global_ability', 0)
            
            # Role counts
            stats['top_count'] += champ.get('is_top', 0)
            stats['jungle_count'] += champ.get('is_jungle', 0)
            stats['mid_count'] += champ.get('is_mid', 0)
            stats['adc_count'] += champ.get('is_adc', 0)
            stats['support_count'] += champ.get('is_support', 0)
            if champ.get('is_adc', 0) == 1 and champ.get('is_ranged', 0) == 1:
                stats['ranged_adc_count'] += 1
        
        return stats
    
    def _role_requirement_score(self, stats: Dict) -> tuple:
        """
        Role Requirements: Only penalize when there are problems.
        If all roles are met with good distribution, no penalty (or bonus).
        """
        bonus = 0.0
        penalty = 0.0
        
        # Check if all required roles are present
        has_all_roles = (
            stats['top_count'] >= 1 and
            stats['jungle_count'] >= 1 and
            stats['mid_count'] >= 1 and
            stats['adc_count'] >= 1 and
            stats['support_count'] >= 1
        )
        
        # CRITICAL: Missing roles - very aggressive penalty
        if not has_all_roles:
            missing_roles = []
            if stats['top_count'] == 0:
                missing_roles.append('Top')
            if stats['jungle_count'] == 0:
                missing_roles.append('Jungle')
            if stats['mid_count'] == 0:
                missing_roles.append('Mid')
            if stats['adc_count'] == 0:
                missing_roles.append('ADC')
            if stats['support_count'] == 0:
                missing_roles.append('Support')
            
            penalty += 0.30 * len(missing_roles)  # -0.30 per missing role
        
        # Only apply additional penalties if roles are present but distribution is poor
        if has_all_roles:
            # Penalize excessive role overlap (too many flex picks competing for same role)
            # Flex picks are fine, but 3+ champs for one role is excessive
            if stats['top_count'] >= 3:
                penalty += 0.05
            if stats['jungle_count'] >= 3:
                penalty += 0.05
            if stats['mid_count'] >= 3:
                penalty += 0.05
            # ADC and Support are more specific - penalize at 2+
            if stats['adc_count'] > 2:
                penalty += 0.08  # ADC is very specific
            if stats['support_count'] >  2:
                penalty += 0.08  # Support is very specific
            
            # Bonus for ideal role distribution (1 of each, with some flex options)
            ideal_distribution = (
                stats['top_count'] >= 1 and stats['top_count'] <= 2 and
                stats['jungle_count'] >= 1 and stats['jungle_count'] <= 2 and
                stats['mid_count'] >= 1 and stats['mid_count'] <= 2 and
                stats['adc_count'] <= 2 and
                stats['support_count'] <= 2
            )
            if ideal_distribution:
                bonus = 0.08  # Perfect role distribution with flex options
        
        # ADC must be ranged (applies regardless of other checks)
        if stats['adc_count'] > 0 and stats['ranged_adc_count'] == 0:
            penalty += 0.20  # ADC not ranged = major issue
        
        return bonus, penalty
    
    def _resource_allocation_score(self, stats: Dict) -> tuple:
        """Resource Allocation: Too many resource-hungry champs"""
        bonus = 0.0
        penalty = 0.0
        
        # Too many scaling/hypercarry champs compete for resources
        resource_hungry = stats['scaling_count'] + stats['hypercarry_count']
        
        if resource_hungry >= 4:
            penalty = 0.08  # Too many champs need gold/XP
        elif resource_hungry == 3:
            penalty = 0.04
        
        return bonus, penalty
    
    def _win_condition_clarity_score(self, stats: Dict) -> tuple:
        """Win Condition: Conflicting strategies"""
        bonus = 0.0
        penalty = 0.0
        
        # Conflicting win conditions
        has_split_push = stats['split_pusher_count'] > 0
        has_team_fight = stats['team_fight_count'] >= 3
        has_pick = stats['pick_count'] >= 2
        
        strategies = sum([has_split_push, has_team_fight, has_pick])
        
        if strategies >= 3:
            penalty = 0.08  # Too many conflicting strategies
        elif strategies == 2 and stats['split_pusher_count'] >= 2 and stats['team_fight_count'] >= 3:
            penalty = 0.06  # Split push + team fight conflict
        
        # Clear win condition is good
        if strategies == 1:
            bonus = 0.04
        
        return bonus, penalty
    
    def _wave_clear_score(self, stats: Dict) -> tuple:
        """Wave clear: More aggressive penalties"""
        bonus = 0.0
        penalty = 0.0
        
        if stats['wave_clear'] < 1:
            penalty = 0.15  # No wave clear = can't control map
        elif stats['wave_clear'] == 1:
            penalty = 0.06  # Weak wave clear
        else:
            bonus = min(stats['wave_clear'] * 0.01, 0.03)
        
        return bonus, penalty
    
    def _frontline_score(self, stats: Dict) -> tuple:
        """Frontline: More aggressive penalties"""
        bonus = 0.0
        penalty = 0.0
        
        has_frontline = stats['frontline_count'] > 1
        has_bruiser_front = stats['bruiser_front_count'] > 2
        
        if has_frontline or has_bruiser_front:
            bonus = 0.04  # Slightly increased
        
        # MUCH more severe penalties
        if stats['bruiser_total'] < 2 and stats['frontline_count'] < 1:
            penalty = 0.20  # No frontline = team gets deleted
        elif stats['frontline_count'] < 1:
            penalty = 0.12  # Weak frontline
        
        return bonus, penalty
    
    def _damage_mix_score(self, stats: Dict) -> tuple:
        """Damage Mix: More aggressive penalties"""
        bonus = 0.0
        penalty = 0.0
        
        # Check for bonus conditions
        if stats['ad_count'] < 4 or stats['ap_count'] < 4:
            bonus = 0.04
        elif stats['mixed_count'] > 1 and stats['ad_count'] > 1 and stats['ap_count'] > 1:
            bonus = 0.04
        
        # Check for penalty conditions - more aggressive
        if stats['ad_count'] > 4 or stats['ap_count'] > 4:
            penalty = 0.10  # Too much of one damage type
        elif stats['ad_count'] == 5 or stats['ap_count'] == 5:
            penalty = 0.15  # All one damage type = very bad
        
        return bonus, penalty
    
    def _poke_score(self, stats: Dict) -> tuple:
        """Poke: More aggressive penalties"""
        bonus = 0.0
        penalty = 0.0
        
        if 2 <= stats['poke_count'] <= 3:
            bonus = 0.03
        elif stats['poke_count'] < 2:
            penalty = 0.08  # No poke = limited options
        elif stats['poke_count'] > 3:
            penalty = 0.05  # Too much poke
        
        return bonus, penalty
    
    def _scaling_score(self, stats: Dict) -> tuple:
        """Scaling: More aggressive penalties"""
        bonus = 0.0
        penalty = 0.0
        
        if stats['scaling_count'] <= 2:
            bonus = min(stats['scaling_count'] * 0.01, 0.02)
        elif stats['scaling_count'] > 3:
            penalty = 0.08  # Too many scaling = weak early
        else:
            penalty = 0.04
        
        return bonus, penalty
    
    def _pick_score(self, stats: Dict) -> tuple:
        """Pick: More aggressive penalties"""
        bonus = 0.0
        penalty = 0.0
        
        has_pick_diver = stats['pick_count'] == 1 and (stats['diver_count'] >= 1 or stats['frontline_count'] >= 1)
        has_two_pick = stats['pick_count'] >= 2
        has_two_diver = stats['diver_count'] >= 2
        
        if has_pick_diver or has_two_pick or has_two_diver:
            bonus = 0.03
        elif stats['pick_count'] == 0 and stats['diver_count'] == 0:
            penalty = 0.06  # No pick potential
        
        return bonus, penalty
    
    def _aoe_combo_score(self, stats: Dict) -> tuple:
        """AOE Combo: More aggressive penalties"""
        bonus = 0.0
        penalty = 0.0
        
        if stats['aoe_combo_count'] == 2:
            bonus = 0.03
        elif stats['aoe_combo_count'] >= 3:
            bonus = 0.04
        elif stats['aoe_combo_count'] == 0:
            penalty = 0.04  # No AOE = limited teamfight
        
        return bonus, penalty
    
    def _team_fight_score(self, stats: Dict) -> tuple:
        """Team Fight: More aggressive penalties"""
        bonus = 0.0
        penalty = 0.0
        
        if stats['team_fight_count'] > 3:
            bonus = 0.04
        elif stats['team_fight_count'] < 3:
            penalty = 0.08  # Weak teamfight
        
        return bonus, penalty
    
    def _split_pusher_score(self, stats: Dict) -> tuple:
        """Split pusher: More aggressive penalties"""
        bonus = 0.0
        penalty = 0.0
        
        if 1 <= stats['split_pusher_count'] <= 2:
            bonus = 0.02
        elif stats['split_pusher_count'] > 2:
            penalty = 0.10  # Too many split pushers
        
        return bonus, penalty
    
    def _hypercarry_peel_score(self, stats: Dict) -> tuple:
        """HyperCarry/Peel: More aggressive penalties"""
        bonus = 0.0
        penalty = 0.0
        
        # Bonus conditions
        if stats['hypercarry_count'] == 1 and stats['peel_count'] == 1:
            bonus = 0.02
        elif stats['hypercarry_count'] >= 2:
            bonus = 0.03
        
        # Penalty conditions - more aggressive
        if stats['hypercarry_count'] > 3 or stats['peel_count'] > 3:
            penalty = 0.10
        elif stats['hypercarry_count'] > 0 and stats['peel_count'] == 0:
            penalty = 0.08  # Hypercarry without peel = vulnerable
        
        return bonus, penalty
    
    def _global_score(self, stats: Dict) -> tuple:
        """Global: More aggressive bonuses"""
        bonus = 0.0
        penalty = 0.0
        
        bonus = min(stats['global_count'] * 0.015, 0.04)  # Slightly increased
        
        return bonus, penalty
    
    def parse_match_url(self, match_url: str) -> Dict:
        """
        Parse a match URL to extract team compositions and determine blue/red side.
        
        Supports:
        - u.gg match URLs (e.g., https://u.gg/lol/match/na1/1234567890)
        - op.gg match URLs
        - u.gg profile/match history pages (will try to extract match data)
        - Riot match IDs (if format is known)
        
        Args:
            match_url: URL to the match page or match history page
            
        Returns:
            Dictionary with 'blue_team' and 'red_team' lists of champion names,
            and 'roles' mapping champions to their roles
        """
        # Normalize URL
        match_url = match_url.strip()
        
        # Try to extract match ID from URL
        match_id = None
        region = None
        
        # Parse u.gg match URLs (e.g., https://u.gg/lol/match/na1/1234567890)
        ugg_match_pattern = r'u\.gg.*?match/([a-z0-9]+)/(\d+)'
        ugg_match = re.search(ugg_match_pattern, match_url, re.IGNORECASE)
        if ugg_match:
            region = ugg_match.group(1)
            match_id = ugg_match.group(2)
            return self._fetch_ugg_match_data(region, match_id)
        
        # Try to parse u.gg profile/match history pages
        # Extract region and summoner info, then try to get match data from the page
        ugg_profile_pattern = r'u\.gg.*?profile/([a-z0-9]+)/([^/]+)'
        ugg_profile_match = re.search(ugg_profile_pattern, match_url, re.IGNORECASE)
        if ugg_profile_match:
            region = ugg_profile_match.group(1)
            # Try to extract match data from the match history page
            return self._fetch_ugg_match_history_data(match_url, region)
        
        # Parse op.gg URLs
        opgg_pattern = r'op\.gg.*?match/([a-z]+)/(\d+)'
        opgg_match = re.search(opgg_pattern, match_url, re.IGNORECASE)
        if opgg_match:
            region = opgg_match.group(1)
            match_id = opgg_match.group(2)
            return self._fetch_opgg_match_data(region, match_id)
        
        # Try direct match ID extraction
        match_id_match = re.search(r'/(\d{10,})', match_url)
        if match_id_match:
            match_id = match_id_match.group(1)
            # Try to infer region from URL or default to common regions
            region_match = re.search(r'/([a-z]{2,3})/', match_url, re.IGNORECASE)
            region = region_match.group(1) if region_match else 'na1'
            return self._fetch_ugg_match_data(region, match_id)
        
        raise ValueError(
            f"Could not parse match URL: {match_url}. "
            f"Supported formats: u.gg match URLs, op.gg match URLs, or u.gg profile pages. "
            f"Alternatively, use calculate_matchup_manual() with team compositions."
        )
    
    def _fetch_ugg_match_history_data(self, match_history_url: str, region: str) -> Dict:
        """
        Try to extract match data from a u.gg match history/profile page.
        Since matches load dynamically via JavaScript, we try to find match data in the page.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(match_history_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            blue_team = []
            red_team = []
            roles = {}
            
            # Strategy 1: Look for JSON data in script tags that might contain match data
            scripts = soup.find_all('script', type='application/json')
            for script in scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    # Recursively search for match data
                    match_data = self._extract_match_data_from_json(data)
                    if match_data and 'blue_team' in match_data and len(match_data['blue_team']) == 5:
                        return match_data
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            # Strategy 2: Look for match history items and try to extract first/recent match
            # This is a fallback - may not work if matches are loaded dynamically
            match_items = soup.find_all(['div', 'tr'], class_=re.compile(r'match|game', re.I))
            # Try to extract champion names from match items
            
            return {
                'blue_team': blue_team,
                'red_team': red_team,
                'roles': roles,
                'url': match_history_url,
                'note': 'Could not automatically extract match data. Please use calculate_matchup_manual() with team compositions.'
            }
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch match history data from u.gg: {e}")
    
    def _extract_match_data_from_json(self, data, match_data=None):
        """Recursively extract match data (teams, champions) from JSON structure."""
        if match_data is None:
            match_data = {'blue_team': [], 'red_team': [], 'roles': {}}
        
        if isinstance(data, dict):
            # Look for team data
            if 'teams' in data or 'participants' in data:
                teams = data.get('teams', [])
                participants = data.get('participants', [])
                
                if participants and len(participants) == 10:
                    # Standard: first 5 are one team, last 5 are other team
                    blue_champs = []
                    red_champs = []
                    
                    for i, participant in enumerate(participants[:5]):
                        champ = participant.get('championName') or participant.get('champion') or participant.get('champ')
                        if champ:
                            blue_champs.append(champ)
                    
                    for i, participant in enumerate(participants[5:]):
                        champ = participant.get('championName') or participant.get('champion') or participant.get('champ')
                        if champ:
                            red_champs.append(champ)
                    
                    if len(blue_champs) == 5 and len(red_champs) == 5:
                        match_data['blue_team'] = blue_champs
                        match_data['red_team'] = red_champs
                        return match_data
            
            # Recursively search nested structures
            for value in data.values():
                result = self._extract_match_data_from_json(value, match_data)
                if result and len(result.get('blue_team', [])) == 5:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._extract_match_data_from_json(item, match_data)
                if result and len(result.get('blue_team', [])) == 5:
                    return result
        
        return match_data if len(match_data.get('blue_team', [])) == 5 else None
    
    def _fetch_ugg_match_data(self, region: str, match_id: str) -> Dict:
        """Fetch match data from u.gg by scraping the match page."""
        url = f"https://u.gg/lol/match/{region}/{match_id}"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            blue_team = []
            red_team = []
            blue_players = []
            red_players = []
            roles = {}
            
            # Try multiple strategies to find match data
            # Strategy 1: Look for JSON data in script tags
            scripts = soup.find_all('script', type='application/json')
            for script in scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    # Try to extract complete match data (champions and players)
                    match_data = self._extract_complete_match_data_from_json(data)
                    if match_data:
                        if match_data.get('blue_team') and len(match_data['blue_team']) == 5:
                            blue_team = match_data['blue_team']
                            red_team = match_data.get('red_team', [])
                        if match_data.get('blue_players') and len(match_data['blue_players']) == 5:
                            blue_players = match_data['blue_players']
                            red_players = match_data.get('red_players', [])
                        if blue_team and blue_players:
                            break
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            # Strategy 2: Extract champions from JSON (fallback)
            if not blue_team:
                for script in scripts:
                    try:
                        import json
                        data = json.loads(script.string)
                        champions = self._extract_champions_from_json(data)
                        if champions and len(champions) == 10:
                            blue_team = champions[:5]
                            red_team = champions[5:]
                            break
                    except (json.JSONDecodeError, AttributeError):
                        continue
            
            # Strategy 3: Look for champion name elements
            if not blue_team:
                # Common class patterns for champion names
                champ_selectors = [
                    {'class': re.compile(r'champion', re.I)},
                    {'data-champion': True},
                    {'data-champ': True},
                    {'class': re.compile(r'champ-name', re.I)},
                ]
                
                for selector in champ_selectors:
                    elements = soup.find_all(attrs=selector)
                    if elements:
                        champs = []
                        for elem in elements:
                            # Try to get champion name from various attributes
                            champ_name = (elem.get('data-champion') or 
                                        elem.get('data-champ') or 
                                        elem.get('title') or
                                        elem.get_text(strip=True))
                            if champ_name and len(champ_name) > 1:
                                champs.append(champ_name)
                        
                        if len(champs) >= 10:
                            blue_team = champs[:5]
                            red_team = champs[5:]
                            break
            
            # Strategy 4: Look for team sections
            if not blue_team:
                team_sections = soup.find_all(['div', 'section'], class_=re.compile(r'team|side', re.I))
                if len(team_sections) >= 2:
                    for team_section in team_sections[:2]:
                        champ_names = []
                        # Look for champion names within team section
                        for elem in team_section.find_all(['div', 'span', 'a']):
                            text = elem.get_text(strip=True)
                            # Check if it looks like a champion name (capitalized, reasonable length)
                            if text and 3 <= len(text) <= 20 and text[0].isupper():
                                champ_names.append(text)
                        
                        if champ_names:
                            if not blue_team:
                                blue_team = champ_names[:5]
                            elif not red_team:
                                red_team = champ_names[:5]
            
            return {
                'blue_team': blue_team,
                'red_team': red_team,
                'blue_players': blue_players,
                'red_players': red_players,
                'roles': roles,
                'match_id': match_id,
                'region': region,
                'url': url
            }
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch match data from u.gg: {e}")
    
    def _extract_complete_match_data_from_json(self, data, match_data=None):
        """Extract complete match data including champions and player names from JSON."""
        if match_data is None:
            match_data = {
                'blue_team': [],
                'red_team': [],
                'blue_players': [],
                'red_players': [],
                'roles': {}
            }
        
        if isinstance(data, dict):
            # Look for participants/players data
            if 'participants' in data:
                participants = data['participants']
                if isinstance(participants, list) and len(participants) == 10:
                    blue_champs = []
                    red_champs = []
                    blue_players = []
                    red_players = []
                    
                    for i, participant in enumerate(participants[:5]):
                        champ = (participant.get('championName') or 
                                participant.get('champion') or 
                                participant.get('champ'))
                        player = (participant.get('summonerName') or 
                                 participant.get('playerName') or 
                                 participant.get('name') or
                                 participant.get('riotId') or
                                 participant.get('gameName'))
                        if champ:
                            blue_champs.append(champ)
                        if player:
                            blue_players.append(player)
                    
                    for i, participant in enumerate(participants[5:]):
                        champ = (participant.get('championName') or 
                                participant.get('champion') or 
                                participant.get('champ'))
                        player = (participant.get('summonerName') or 
                                 participant.get('playerName') or 
                                 participant.get('name') or
                                 participant.get('riotId') or
                                 participant.get('gameName'))
                        if champ:
                            red_champs.append(champ)
                        if player:
                            red_players.append(player)
                    
                    if len(blue_champs) == 5 and len(red_champs) == 5:
                        match_data['blue_team'] = blue_champs
                        match_data['red_team'] = red_champs
                    if len(blue_players) == 5 and len(red_players) == 5:
                        match_data['blue_players'] = blue_players
                        match_data['red_players'] = red_players
                    
                    if len(match_data['blue_team']) == 5:
                        return match_data
            
            # Recursively search nested structures
            for value in data.values():
                result = self._extract_complete_match_data_from_json(value, match_data)
                if result and len(result.get('blue_team', [])) == 5:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._extract_complete_match_data_from_json(item, match_data)
                if result and len(result.get('blue_team', [])) == 5:
                    return result
        
        return match_data if len(match_data.get('blue_team', [])) == 5 else None
    
    def _extract_champions_from_json(self, data, champions=None):
        """Recursively extract champion names from JSON data."""
        if champions is None:
            champions = []
        
        if isinstance(data, dict):
            # Look for common keys that might contain champion names
            for key in ['champion', 'championName', 'champ', 'name']:
                if key in data and isinstance(data[key], str):
                    champ_name = data[key]
                    if champ_name and champ_name not in champions:
                        champions.append(champ_name)
            
            # Recursively search nested structures
            for value in data.values():
                self._extract_champions_from_json(value, champions)
        elif isinstance(data, list):
            for item in data:
                self._extract_champions_from_json(item, champions)
        
        return champions
    
    def _fetch_opgg_match_data(self, region: str, match_id: str) -> Dict:
        """Fetch match data from op.gg by scraping the match page."""
        url = f"https://www.op.gg/matches/{region}/{match_id}"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            blue_team = []
            red_team = []
            roles = {}
            
            # Similar parsing logic for op.gg
            # Placeholder structure
            return {
                'blue_team': blue_team,
                'red_team': red_team,
                'roles': roles,
                'match_id': match_id,
                'region': region,
                'url': url
            }
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch match data from op.gg: {e}")
    
    def get_champion_matchup_winrate(self, champion1: str, champion2: str, role: str = None) -> Optional[float]:
        """
        Get win rate of champion1 vs champion2 from u.gg.
        Uses caching to avoid redundant API calls. If we have champ1 vs champ2,
        we can derive champ2 vs champ1 as (1 - winrate).
        
        Args:
            champion1: First champion name
            champion2: Opposing champion name
            role: Optional role filter (top, jungle, mid, adc, support)
            
        Returns:
            Win rate as a decimal (e.g., 0.52 for 52%), or None if unavailable
        """
        # Normalize champion names (for URL - u.gg uses "monkeyking" not "wukong")
        champ1_normalized = self._normalize_champion_name(champion1)
        champ2_normalized = self._normalize_champion_name(champion2)
        
        # Keep original names for display
        champ1_display = champion1
        champ2_display = champion2
        
        # Create cache key (always use alphabetical order to ensure consistency)
        # This way (Ashe, Varus) and (Varus, Ashe) use the same cache entry
        # Store as (min_champ, max_champ, role) -> winrate of min_champ vs max_champ
        sorted_champs = sorted([champ1_normalized, champ2_normalized])
        cache_key = (sorted_champs[0], sorted_champs[1], role)
        
        # Check cache - if we have it, return the correct winrate
        if cache_key in self.matchup_cache:
            cached_winrate = self.matchup_cache[cache_key]
            if cached_winrate is None:
                return None
            # Return cached winrate if champ1 is first alphabetically, otherwise return 1 - winrate
            if champ1_normalized == sorted_champs[0]:
                return cached_winrate
            else:
                return 1.0 - cached_winrate
        
        # Need to make API call - build u.gg matchup URL
        # Use direct matchup URL: /champions/{champ}/build?opp={opponent}
        # This is simpler and more reliable than scraping the counter page
        url = f"https://u.gg/lol/champions/{champ1_normalized}/build?opp={champ2_normalized}"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug mode disabled by default (set to True to enable debugging)
            debug_mode = False
            # Uncomment the line below to enable debug for specific matchups:
            # debug_mode = (len(self.matchup_cache) < 5 or 
            #              ('wukong' in champ1_normalized.lower() and 'xinzhao' in champ2_normalized.lower()) or
            #              ('xinzhao' in champ1_normalized.lower() and 'wukong' in champ2_normalized.lower()) or
            #              ('ryze' in champ1_normalized.lower() and 'taliyah' in champ2_normalized.lower()) or
            #              ('taliyah' in champ1_normalized.lower() and 'ryze' in champ2_normalized.lower()))
            if debug_mode:
                print(f"\n[DEBUG] Fetching matchup data for {champ1_display} vs {champ2_display} (role: {role})")
                print(f"[DEBUG] Using u.gg URL: {url}")
            
            # Strategy 1: Look for win rate directly in the page HTML
            # The direct matchup page shows "50.53% Win Rate" prominently
            # Try multiple approaches to find it
            winrate = None
            
            # Approach 1: Look for text pattern "XX.XX% Win Rate"
            winrate_text = soup.find(text=re.compile(r'\d+\.?\d*\s*%\s*Win\s*Rate', re.I))
            if winrate_text:
                match = re.search(r'(\d+\.?\d*)', str(winrate_text))
                if match:
                    winrate_pct = float(match.group(1))
                    winrate = winrate_pct / 100.0
                    if debug_mode:
                        print(f"[DEBUG] Found win rate via text pattern: {winrate_pct}%")
            
            # Approach 2: Look for elements containing "Win Rate" and find nearby percentage
            if winrate is None:
                win_rate_elements = soup.find_all(string=re.compile(r'Win\s*Rate', re.I))
                for elem in win_rate_elements:
                    # Look in parent and siblings for percentage
                    parent = elem.parent if hasattr(elem, 'parent') else None
                    if parent:
                        # Check parent text
                        parent_text = parent.get_text()
                        match = re.search(r'(\d+\.?\d*)\s*%\s*Win\s*Rate', parent_text, re.I)
                        if match:
                            winrate_pct = float(match.group(1))
                            winrate = winrate_pct / 100.0
                            if debug_mode:
                                print(f"[DEBUG] Found win rate in parent element: {winrate_pct}%")
                            break
                        # Check next sibling
                        if hasattr(parent, 'next_sibling') and parent.next_sibling:
                            sibling_text = parent.next_sibling.get_text() if hasattr(parent.next_sibling, 'get_text') else str(parent.next_sibling)
                            match = re.search(r'(\d+\.?\d*)%', sibling_text)
                            if match:
                                winrate_pct = float(match.group(1))
                                winrate = winrate_pct / 100.0
                                if debug_mode:
                                    print(f"[DEBUG] Found win rate in sibling: {winrate_pct}%")
                                break
            
            # Approach 3: Search all text for pattern "XX.XX%" near "Win Rate" or "vs. Xin Zhao"
            if winrate is None:
                all_text = soup.get_text()
                # Look for pattern like "50.53% Win Rate" or "50.53%" near opponent name
                patterns = [
                    r'(\d+\.?\d*)\s*%\s*Win\s*Rate',
                    r'Win\s*Rate[:\s]*(\d+\.?\d*)%',
                    rf'vs\.?\s*{re.escape(champ2_display)}[^\d]*(\d+\.?\d*)%',
                ]
                for pattern in patterns:
                    match = re.search(pattern, all_text, re.I)
                    if match:
                        winrate_pct = float(match.group(1))
                        winrate = winrate_pct / 100.0
                        if debug_mode:
                            print(f"[DEBUG] Found win rate via text search: {winrate_pct}%")
                        break
            
            if winrate is not None and 0.25 <= winrate <= 0.75:
                if debug_mode:
                    print(f"[DEBUG] Using win rate from HTML: {winrate:.4f} ({winrate*100:.2f}%)")
                    print(f"[DEBUG] This is {champ1_display}'s winrate vs {champ2_display} (no inversion needed)")
                
                # This is champ1's winrate vs champ2 (already correct, no inversion needed)
                # The page shows "Wukong vs XinZhao: 50.53%" which is Wukong's winrate
                # Cache the result (always store in alphabetical order: min_champ vs max_champ)
                if champ1_normalized == sorted_champs[0]:
                    self.matchup_cache[cache_key] = winrate
                    return winrate
                else:
                    self.matchup_cache[cache_key] = 1.0 - winrate
                    return winrate
            elif winrate is not None:
                if debug_mode:
                    print(f"[DEBUG] Found winrate {winrate:.4f} but outside valid range (0.25-0.75), skipping")
            
            # If HTML parsing didn't work, try to save the page for debugging
            if winrate is None and debug_mode:
                print(f"[DEBUG] HTML parsing failed, trying to find winrate in page structure...")
                # Look for common HTML structures that might contain winrate
                # Try finding elements with class names that might contain winrate
                for tag in soup.find_all(['div', 'span', 'p', 'h1', 'h2', 'h3']):
                    text = tag.get_text() if hasattr(tag, 'get_text') else str(tag)
                    if 'win rate' in text.lower() or 'winrate' in text.lower():
                        match = re.search(r'(\d+\.?\d*)%', text)
                        if match:
                            winrate_pct = float(match.group(1))
                            candidate = winrate_pct / 100.0
                            if 0.25 <= candidate <= 0.75:
                                winrate = candidate
                                print(f"[DEBUG] Found winrate in element text: {winrate_pct}%")
                                break
                
                if winrate is None:
                    print(f"[DEBUG] Still not found in HTML elements, will try JSON fallback...")
                    # Dump a sample of the page text to see what we're working with
                    page_text_sample = soup.get_text()[:500] if hasattr(soup, 'get_text') else str(soup)[:500]
                    print(f"[DEBUG] Page text sample (first 500 chars): {page_text_sample}")
            
            # Strategy 2: Look for JSON data in script tags (fallback)
            # NOTE: Since we're using direct matchup page (?opp=), JSON should also contain champ1's winrate directly
            scripts = soup.find_all('script', type='application/json')
            for script in scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    # Debug mode is already set above (disabled by default)
                    # Reuse the same debug_mode variable
                    if debug_mode:
                        print(f"\n[DEBUG] Extracting matchup data for {champ1_display} vs {champ2_display} (role: {role})")
                        print(f"[DEBUG] Using u.gg URL: {url} (normalized: {champ1_normalized} vs {champ2_normalized})")
                        print(f"[DEBUG] Searching JSON data structure...")
                    # For matching in JSON, try both normalized and original name
                    # JSON might use "Xin Zhao" with space, not "xinzhao"
                    champ2_for_matching = champ2_normalized
                    # Also try with space if it's a camelCase name
                    if not ' ' in champ2_display and not '-' in champ2_display:
                        # Try adding space before capitals (e.g., "XinZhao" -> "Xin Zhao")
                        # Note: re is imported at module level
                        champ2_with_space = re.sub(r'(?<!^)(?<! )([A-Z])', r' \1', champ2_display)
                        if champ2_with_space != champ2_display:
                            # Try matching with space version too
                            champ2_for_matching = champ2_with_space.lower()
                    
                    winrate = self._extract_matchup_winrate_from_json(data, champ2_for_matching, role, debug=debug_mode)
                    # If not found with normalized, try with space version
                    if winrate is None and champ2_for_matching != champ2_normalized:
                        if debug_mode:
                            print(f"[DEBUG] Trying alternative match: '{champ2_normalized}'")
                        winrate = self._extract_matchup_winrate_from_json(data, champ2_normalized, role, debug=debug_mode)
                    if winrate is not None:
                        # NOTE: We're using direct matchup page (?opp=), so winrate should be champ1's winrate directly
                        # No inversion needed - the page shows "Wukong vs XinZhao: 50.53%" which is Wukong's winrate
                        if 0.25 <= winrate <= 0.75:
                            champ1_winrate = winrate
                        else:
                            # Value outside reasonable range, skip it
                            if debug_mode:
                                print(f"[DEBUG] JSON winrate {winrate:.4f} outside valid range (0.25-0.75), skipping")
                            continue
                        
                        if debug_mode:
                            print(f"[DEBUG] Extracted winrate from JSON: {champ1_winrate:.4f} ({champ1_winrate*100:.2f}%)")
                            print(f"[DEBUG] This is {champ1_display}'s winrate vs {champ2_display} (no inversion needed)")
                        
                        # Cache the result (always store in alphabetical order: min_champ vs max_champ)
                        if champ1_normalized == sorted_champs[0]:
                            self.matchup_cache[cache_key] = champ1_winrate
                            return champ1_winrate
                        else:
                            self.matchup_cache[cache_key] = 1.0 - champ1_winrate
                            return champ1_winrate
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            # Strategy 2: Look for matchup table or specific opponent data
            # Find elements that might contain the opponent's name
            opponent_elements = soup.find_all(text=re.compile(champ2_normalized, re.I))
            for elem in opponent_elements:
                parent = elem.parent if hasattr(elem, 'parent') else None
                if parent:
                    # Look for win rate near the opponent name
                    winrate_elem = parent.find_next(text=re.compile(r'\d+\.?\d*\s*%'))
                    if winrate_elem:
                        match = re.search(r'(\d+\.?\d*)', str(winrate_elem))
                        if match:
                            winrate = float(match.group(1)) / 100.0
                            
                            # IMPORTANT: u.gg counter pages show the OPPONENT's counter rate
                            # Invert to get champ1's actual winrate vs champ2
                            champ1_winrate = 1.0 - winrate
                            
                            # Cache the result (always store in alphabetical order: min_champ vs max_champ)
                            if champ1_normalized == sorted_champs[0]:
                                self.matchup_cache[cache_key] = champ1_winrate
                                return champ1_winrate
                            else:
                                self.matchup_cache[cache_key] = 1.0 - champ1_winrate
                                return champ1_winrate
            
            # Strategy 3: Look for win rate patterns in the page
            winrate_pattern = r'(\d+\.?\d*)\s*%'
            all_text = soup.get_text()
            matches = list(re.finditer(winrate_pattern, all_text))
            
            # Try to find win rate near opponent mentions
            for match in matches:
                context_start = max(0, match.start() - 100)
                context_end = min(len(all_text), match.end() + 100)
                context = all_text[context_start:context_end].lower()
                if champ2_normalized in context:
                    winrate = float(match.group(1)) / 100.0
                    
                    # IMPORTANT: u.gg counter pages show the OPPONENT's counter rate
                    # Invert to get champ1's actual winrate vs champ2
                    champ1_winrate = 1.0 - winrate
                    
                    # Cache the result (always store in alphabetical order: min_champ vs max_champ)
                    if champ1_normalized == sorted_champs[0]:
                        self.matchup_cache[cache_key] = champ1_winrate
                        return champ1_winrate
                    else:
                        self.matchup_cache[cache_key] = 1.0 - champ1_winrate
                        return champ1_winrate
            
            # No matchup found - cache None to avoid retrying
            self.matchup_cache[cache_key] = None
            return None
            
        except requests.RequestException:
            # Cache None to avoid retrying failed requests
            self.matchup_cache[cache_key] = None
            return None
        except (ValueError, AttributeError):
            self.matchup_cache[cache_key] = None
            return None
    
    def _normalize_champion_name(self, champion_name: str) -> str:
        """Normalize champion name for URL (e.g., 'Aurelion Sol' -> 'aurelion-sol')."""
        # Handle special cases
        # NOTE: u.gg uses "wukong" not "monkeyking" in URLs (verified: https://u.gg/lol/champions/wukong/build)
        special_cases = {
            'nunu': 'nunu-willump',
            'renata glasc': 'renata',
        }
        
        champ_lower = champion_name.lower().strip()
        if champ_lower in special_cases:
            return special_cases[champ_lower]
        
        # Standard normalization
        normalized = champ_lower.replace("'", "").replace(" ", "-")
        return normalized
    
    def _normalize_champion_name_for_ugg(self, champion_name: str) -> str:
        """
        Normalize champion name for u.gg matching.
        Handles Riot API format (e.g., 'JarvanIV' -> 'Jarvan IV').
        """
        # Riot API returns names like "JarvanIV", "TwistedFate", "Jarvan IV"
        # u.gg uses "Jarvan IV", "Twisted Fate", etc.
        
        # If it's already in correct format, return as-is
        if ' ' in champion_name:
            return champion_name
        
        # Handle camelCase names (e.g., "JarvanIV" -> "Jarvan IV")
        # Insert space before capital letters (but not the first one)
        import re
        normalized = re.sub(r'(?<!^)(?<! )([A-Z])', r' \1', champion_name)
        
        # Handle special cases
        special_cases = {
            'Jarvan IV': 'Jarvan IV',  # Already correct
            'JarvanIV': 'Jarvan IV',
            'Twisted Fate': 'Twisted Fate',
            'TwistedFate': 'Twisted Fate',
            'Miss Fortune': 'Miss Fortune',
            'MissFortune': 'Miss Fortune',
            'Master Yi': 'Master Yi',
            'MasterYi': 'Master Yi',
            'Dr Mundo': 'Dr. Mundo',
            'DrMundo': 'Dr. Mundo',
            'Lee Sin': 'Lee Sin',
            'LeeSin': 'Lee Sin',
        }
        
        if normalized in special_cases:
            return special_cases[normalized]
        
        return normalized
    
    def _extract_matchup_winrate_from_json(self, data, opponent_name: str, role: str = None, debug: bool = False) -> Optional[float]:
        """Extract matchup win rate from JSON data structure."""
        if isinstance(data, dict):
            # Look for matchup data
            if 'matchups' in data or 'counters' in data:
                matchups = data.get('matchups') or data.get('counters', [])
                if debug:
                    print(f"[DEBUG] Found {len(matchups)} matchups in JSON")
                    if len(matchups) > 0 and isinstance(matchups[0], dict):
                        print(f"[DEBUG] Sample matchup keys: {list(matchups[0].keys())}")
                        # Show first few opponent names for debugging
                        sample_names = []
                        for m in matchups[:5]:
                            if isinstance(m, dict):
                                name = (m.get('opponent') or m.get('champion') or m.get('name', 'N/A'))
                                sample_names.append(name)
                        print(f"[DEBUG] Sample opponent names: {sample_names}")
                
                for matchup in matchups:
                    if isinstance(matchup, dict):
                        opp_name = (matchup.get('opponent') or 
                                  matchup.get('champion') or 
                                  matchup.get('name', '')).lower()
                        if debug and len(matchups) <= 20:  # Only print all if small list
                            print(f"[DEBUG] Checking: '{opp_name}' vs '{opponent_name}'")
                        
                        # Try multiple matching strategies:
                        # 1. Direct substring match (handles "xinzhao" vs "xin zhao")
                        # 2. Remove spaces/hyphens and compare (handles "xin zhao" vs "xinzhao")
                        # 3. Check if normalized versions match
                        opp_name_normalized = opp_name.replace(' ', '').replace('-', '').replace("'", "").replace('.', '')
                        opponent_name_normalized = opponent_name.replace(' ', '').replace('-', '').replace("'", "").replace('.', '')
                        
                        match_found = (
                            opponent_name in opp_name or 
                            opp_name in opponent_name or
                            opponent_name_normalized in opp_name_normalized or
                            opp_name_normalized in opponent_name_normalized or
                            opponent_name_normalized == opp_name_normalized
                        )
                        
                        if match_found:
                            # Check role if specified
                            if role:
                                matchup_role = matchup.get('role', '').lower()
                                if matchup_role != role.lower():
                                    continue
                            
                            # DEBUG: Print all available fields in the matchup dict
                            if debug:
                                print(f"\n[DEBUG] Found matchup for {opponent_name} (role: {role})")
                                print(f"[DEBUG] All fields in matchup dict: {list(matchup.keys())}")
                                print(f"[DEBUG] Full matchup dict: {matchup}")
                            
                            # Extract win rate - try multiple possible field names
                            # u.gg might use different field names for win rate
                            # IMPORTANT: Skip counter_score and difficulty - these are NOT win rates
                            counter_score = matchup.get('counter_score') or matchup.get('difficulty') or matchup.get('score')
                            
                            # Look for actual win rate fields (prefer these over generic 'score' fields)
                            winrate = (matchup.get('winrate') or 
                                     matchup.get('win_rate') or 
                                     matchup.get('winRate') or
                                     matchup.get('matchupWinRate') or
                                     matchup.get('matchup_winrate') or
                                     matchup.get('wr') or
                                     matchup.get('win_rate_percent') or
                                     matchup.get('winRatePercent') or
                                     matchup.get('winRatePercent') or
                                     matchup.get('win_rate_pct'))
                            
                            # If we only have a generic 'score' field and it's not in win rate range, skip it
                            # (it's probably a counter score, not a win rate)
                            if not winrate and counter_score is not None:
                                try:
                                    score_val = float(counter_score) if isinstance(counter_score, (int, float)) else float(str(counter_score).replace('%', ''))
                                    if score_val > 1:
                                        score_val = score_val / 100.0
                                    # Counter scores are typically very low (0.1-0.3) or very high (0.7-0.9)
                                    # Win rates are typically 0.4-0.6. If score is in win rate range, use it
                                    if 0.4 <= score_val <= 0.6:
                                        winrate = counter_score
                                    # Otherwise, it's probably a counter score, not a win rate - skip it
                                except (ValueError, TypeError):
                                    pass
                            
                            if debug:
                                print(f"[DEBUG] Extracted winrate field value: {winrate}")
                                print(f"[DEBUG] Counter score value: {counter_score}")
                            
                            if winrate:
                                if isinstance(winrate, str):
                                    # Remove % if present
                                    winrate = winrate.replace('%', '').strip()
                                try:
                                    wr = float(winrate)
                                    # If it's > 1, assume it's a percentage
                                    if wr > 1:
                                        wr = wr / 100.0
                                    
                                    # Validate: win rates should typically be between 0.4 and 0.6 (40-60%)
                                    # Most matchups are close to 50%, so values far from 0.5 are suspicious
                                    if 0.35 <= wr <= 0.65:
                                        # This looks like a valid win rate
                                        return wr
                                    elif 0.25 <= wr < 0.35 or 0.65 < wr <= 0.75:
                                        # Still plausible but less common - use as-is
                                        return wr
                                    elif wr < 0.25:
                                        # Very low - might be opponent's win rate or counter score
                                        # Try inverting, but only if result is reasonable
                                        inverted = 1.0 - wr
                                        if 0.35 <= inverted <= 0.65:
                                            return inverted
                                        # If inversion doesn't help, this is probably not a win rate - skip it
                                        return None
                                    elif wr > 0.75:
                                        # Very high - might already be inverted or wrong field
                                        inverted = 1.0 - wr
                                        if 0.35 <= inverted <= 0.65:
                                            return inverted
                                        # If inversion doesn't help, this is probably not a win rate - skip it
                                        return None
                                    
                                    # If still outside range, it's probably not a win rate
                                    return None
                                except (ValueError, TypeError):
                                    pass
            
            # Recursively search nested structures (pass debug parameter)
            if debug:
                print(f"[DEBUG] No matchups/counters found in top level, searching nested structures...")
                print(f"[DEBUG] Top-level keys: {list(data.keys())[:10]}")  # Show first 10 keys
            for value in data.values():
                result = self._extract_matchup_winrate_from_json(value, opponent_name, role, debug)
                if result is not None:
                    return result
        elif isinstance(data, list):
            if debug:
                print(f"[DEBUG] Searching list with {len(data)} items...")
            for item in data:
                result = self._extract_matchup_winrate_from_json(item, opponent_name, role, debug)
                if result is not None:
                    return result
        
        return None
    
    def calculate_lane_matchup_score(self, blue_team: List[str], red_team: List[str], 
                                     roles: Dict[str, str] = None, w1: float = 0.35, 
                                     return_details: bool = False) -> float:
        """
        Calculate lane matchup score: w1 * 1/5∑(WRb - WRr) for all 5 roles.
        
        Formula: Sum of (Blue win rate vs Red - Red win rate vs Blue) across all 5 roles,
        then average and multiply by weight factor w1.
        
        Args:
            blue_team: List of 5 blue side champions [top, jungle, mid, adc, support]
            red_team: List of 5 red side champions [top, jungle, mid, adc, support]
            roles: Optional mapping of champion to role (if not provided, assumes standard order)
            w1: Weight factor (default 0.35)
            return_details: If True, returns dict with score and breakdown details
            
        Returns:
            Lane matchup score clamped to [-0.15, 0.15], or dict with details if return_details=True
        """
        if len(blue_team) != 5 or len(red_team) != 5:
            raise ValueError("Both teams must have exactly 5 champions")
        
        # Standard role order if not provided
        role_order = ['top', 'jungle', 'mid', 'adc', 'support']
        
        # Calculate matchup differences for each role: WRb - WRr
        matchup_differences = []
        matchup_details = []
        
        # DEBUG: Track matchup cache status
        debug_matchups = hasattr(self, '_debug_matchups') and self._debug_matchups
        matchup_cache_before = len(self.matchup_cache)
        
        for i, role in enumerate(role_order):
            blue_champ = blue_team[i]
            red_champ = red_team[i]
            
            # If either side hasn't picked this role yet, treat as neutral and skip API calls
            if blue_champ is None or red_champ is None:
                matchup_differences.append(0.0)
                matchup_details.append({
                    'role': role,
                    'blue_champ': blue_champ,
                    'red_champ': red_champ,
                    'wr_blue_vs_red': None,
                    'wr_red_vs_blue': None,
                    'difference': 0.0
                })
                continue
            
            # OPTIMIZATION: Only make ONE API call per matchup pair
            # If we get blue vs red, we can derive red vs blue = 1 - (blue vs red)
            if debug_matchups:
                cache_before = len(self.matchup_cache)
            wr_blue_vs_red = self.get_champion_matchup_winrate(blue_champ, red_champ, role)
            if debug_matchups:
                cache_after = len(self.matchup_cache)
                cache_hit = cache_after == cache_before
                status = "CACHE HIT" if cache_hit else f"NEW API CALL (cache: {cache_before} -> {cache_after})"
                wr_str = f"{wr_blue_vs_red:.2%}" if wr_blue_vs_red is not None else "N/A"
                print(f"        [{role}] {blue_champ} vs {red_champ}: {wr_str} - {status}")
            
            # Derive red vs blue from blue vs red (no additional API call needed)
            if wr_blue_vs_red is not None:
                wr_red_vs_blue = 1.0 - wr_blue_vs_red
                # Calculate difference: WRb - WRr (Blue advantage)
                diff = wr_blue_vs_red - wr_red_vs_blue
                # This simplifies to: wr_blue_vs_red - (1 - wr_blue_vs_red) = 2 * wr_blue_vs_red - 1
                # But we keep it explicit for clarity
                matchup_differences.append(diff)
                matchup_details.append({
                    'role': role,
                    'blue_champ': blue_champ,
                    'red_champ': red_champ,
                    'wr_blue_vs_red': wr_blue_vs_red,
                    'wr_red_vs_blue': wr_red_vs_blue,
                    'difference': diff
                })
            else:
                # No data available, assume neutral matchup
                matchup_differences.append(0.0)
                matchup_details.append({
                    'role': role,
                    'blue_champ': blue_champ,
                    'red_champ': red_champ,
                    'wr_blue_vs_red': None,
                    'wr_red_vs_blue': None,
                    'difference': 0.0
                })
        
        # Calculate average: 1/5∑(WRb - WRr)
        # Sum all differences across 5 roles, then divide by 5
        if matchup_differences:
            sum_differences = sum(matchup_differences)
            avg_difference = sum_differences / len(matchup_differences)
        else:
            avg_difference = 0.0
        
        # Apply weight: w1 * 1/5∑(WRb - WRr)
        raw_score = w1 * avg_difference
        
        # Clamp to [-0.15, 0.15]
        clamped_score = max(-0.15, min(0.15, raw_score))
        
        if return_details:
            return {
                'score': clamped_score,
                'raw_score': raw_score,
                'sum_differences': sum_differences,
                'avg_difference': avg_difference,
                'w1': w1,
                'matchup_details': matchup_details
            }
        
        return clamped_score
    
    def calculate_matchup_from_url(self, match_url: str, w1: float = 0.35) -> Dict:
        """
        Calculate lane matchup score from a match URL.
        
        Args:
            match_url: URL to the match (u.gg, op.gg, etc.)
            w1: Weight factor for lane matchups (default 0.35)
            
        Returns:
            Dictionary with matchup score and details
        """
        # Parse match URL to get team compositions
        match_data = self.parse_match_url(match_url)
        
        blue_team = match_data['blue_team']
        red_team = match_data['red_team']
        roles = match_data.get('roles', {})
        
        # If teams are empty (couldn't parse automatically), raise error
        if not blue_team or not red_team:
            raise ValueError(
                f"Could not automatically extract team compositions from URL. "
                f"Please provide teams manually using calculate_lane_matchup_score() or check the URL format."
            )
        
        # Calculate lane matchup score
        matchup_score = self.calculate_lane_matchup_score(blue_team, red_team, roles, w1)
        
        return {
            'match_url': match_url,
            'blue_team': blue_team,
            'red_team': red_team,
            'matchup_score': matchup_score,
            'w1': w1
        }
    
    def calculate_matchup_manual(self, blue_team: List[str], red_team: List[str], 
                                  w1: float = 0.35, return_details: bool = True) -> Dict:
        """
        Calculate lane matchup score with manually provided teams.
        
        Args:
            blue_team: List of 5 blue side champions [top, jungle, mid, adc, support]
            red_team: List of 5 red side champions [top, jungle, mid, adc, support]
            w1: Weight factor for lane matchups (default 0.35)
            return_details: If True, includes detailed breakdown of all matchups
            
        Returns:
            Dictionary with matchup score and details
        """
        if return_details:
            details = self.calculate_lane_matchup_score(blue_team, red_team, None, w1, return_details=True)
            return {
                'blue_team': blue_team,
                'red_team': red_team,
                'matchup_score': details['score'],
                'w1': w1,
                'sum_differences': details['sum_differences'],
                'avg_difference': details['avg_difference'],
                'matchup_details': details['matchup_details']
            }
        else:
            matchup_score = self.calculate_lane_matchup_score(blue_team, red_team, None, w1)
            return {
                'blue_team': blue_team,
                'red_team': red_team,
                'matchup_score': matchup_score,
                'w1': w1
            }
    
    def get_player_champion_winrate_from_riot_api(self, player_name: str, champion: str, 
                                                    region: str = 'americas', 
                                                    min_games: int = 10, 
                                                    debug: bool = False,
                                                    queue_ids: List[int] = None) -> Optional[Dict]:
        """
        Get a player's win rate on a specific champion from Riot API (ranked matches only).
        
        Args:
            player_name: Riot ID (e.g., 'dtboss#2003')
            champion: Champion name (e.g., 'Renekton')
            region: API region (default 'americas')
            min_games: Minimum games required (default 10)
            debug: If True, print debug information
            queue_ids: List of queue IDs to filter (default [420, 440] for ranked)
            
        Returns:
            Dictionary with 'winrate' (float), 'games_played' (int), or None if not found
        """
        if not self.riot_api_key:
            if debug:
                print("  [DEBUG] No Riot API key available, skipping API method")
            return None
        
        if queue_ids is None:
            queue_ids = [420]  # 420 = Ranked Solo/Duo
        
        # Parse Riot ID
        if '#' not in player_name:
            if debug:
                print(f"  [DEBUG] Player name '{player_name}' is not in Riot ID format (gameName#tagLine)")
            return None
        
        game_name, tag_line = player_name.split('#', 1)
        
        try:
            # Get PUUID
            if debug:
                print(f"  [DEBUG] Getting PUUID for {game_name}#{tag_line}...")
            puuid = self.get_puuid_from_riot_id(game_name, tag_line, region)
            
            # Normalize champion name for comparison (Riot API uses camelCase like "JarvanIV")
            champion_normalized = champion.strip()
            
            # Track wins and losses for this champion
            wins = 0
            losses = 0
            max_matches_to_check = 100  # Check up to 100 matches
            
            # Check each queue ID
            for queue_id in queue_ids:
                if debug:
                    queue_name = "Ranked Solo/Duo" if queue_id == 420 else "Ranked Flex" if queue_id == 440 else f"Queue {queue_id}"
                    print(f"  [DEBUG] Fetching {queue_name} matches (queue_id={queue_id})...")
                
                # Get match history for this queue
                match_ids = self.get_match_history_by_puuid(
                    puuid, 
                    region=region, 
                    start=0, 
                    count=max_matches_to_check,
                    queue_id=queue_id
                )
                
                if debug:
                    print(f"  [DEBUG] Found {len(match_ids)} matches for queue {queue_id}")
                
                # Process matches
                for match_id in match_ids:
                    try:
                        # Get the raw match data from API to find our player
                        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
                        headers = {'X-Riot-Token': self.riot_api_key}
                        response = requests.get(url, headers=headers, timeout=10)
                        response.raise_for_status()
                        match_json = response.json()
                        
                        # Find our player in the match
                        info = match_json.get('info', {})
                        participants = info.get('participants', [])
                        
                        for participant in participants:
                            if participant.get('puuid') == puuid:
                                # Found our player
                                match_champion = participant.get('championName', '')
                                won = participant.get('win', False)
                                
                                # Check if this is the champion we're looking for
                                if match_champion == champion_normalized:
                                    if won:
                                        wins += 1
                                    else:
                                        losses += 1
                                    
                                    if debug:
                                        result = "WIN" if won else "LOSS"
                                        print(f"  [DEBUG] Match {match_id}: {match_champion} - {result} (Total: {wins}W {losses}L)")
                                
                                break  # Found our player, no need to check other participants
                        
                    except Exception as e:
                        if debug:
                            print(f"  [DEBUG] Error processing match {match_id}: {e}")
                        continue
                    
                    # Stop if we have enough games
                    if (wins + losses) >= max_matches_to_check:
                        break
                
                # Stop if we have enough games
                if (wins + losses) >= max_matches_to_check:
                    break
            
            games_played = wins + losses
            
            if debug:
                print(f"  [DEBUG] Final stats: {wins}W {losses}L = {games_played} games on {champion}")
            
            if games_played == 0:
                if debug:
                    print(f"  [DEBUG] No games found for {champion}")
                return None
            
            winrate = wins / games_played if games_played > 0 else 0.5
            
            # Return winrate (use 0.5 if not enough games, but still return the actual stats)
            if games_played >= min_games:
                return {
                    'winrate': winrate,
                    'games_played': games_played,
                    'wins': wins,
                    'losses': losses
                }
            else:
                return {
                    'winrate': 0.5,  # Default if not enough games
                    'games_played': games_played,
                    'wins': wins,
                    'losses': losses
                }
                
        except Exception as e:
            if debug:
                print(f"  [DEBUG] Error in Riot API method: {e}")
            return None
    
    def get_player_champion_winrate(self, player_name: str, champion: str, region: str = 'na1', 
                                     min_games: int = 10, debug: bool = False, 
                                     use_riot_api: bool = True) -> Optional[Dict]:
        """
        Get a player's win rate on a specific champion.
        Tries Riot API first (if available and use_riot_api=True), then falls back to u.gg scraping.
        
        Args:
            player_name: Summoner name or Riot ID (e.g., 'dtboss-2003' or 'dtboss#2003')
            champion: Champion name
            region: Region code (default 'na1' for u.gg, 'americas' for Riot API)
            min_games: Minimum games required (default 10)
            debug: If True, print debug information
            use_riot_api: If True, try Riot API first (default True)
            
        Returns:
            Dictionary with 'winrate' (float), 'games_played' (int), or None if not found
        """
        # Try Riot API first if available and requested
        if use_riot_api and self.riot_api_key:
            # Convert region code for Riot API (na1 -> americas, etc.)
            api_region_map = {
                'na1': 'americas', 'euw1': 'europe', 'eun1': 'europe',
                'kr': 'asia', 'jp1': 'asia', 'br1': 'americas',
                'la1': 'americas', 'la2': 'americas', 'oc1': 'americas',
                'tr1': 'europe', 'ru': 'europe'
            }
            api_region = api_region_map.get(region.lower(), 'americas')
            
            if debug:
                print(f"  [DEBUG] Trying Riot API first...")
            
            result = self.get_player_champion_winrate_from_riot_api(
                player_name, champion, region=api_region, min_games=min_games, debug=debug
            )
            
            if result:
                if debug:
                    print(f"  [DEBUG] ✓ Successfully retrieved from Riot API")
                return result
            elif debug:
                print(f"  [DEBUG] Riot API method failed, falling back to u.gg scraping...")
        
        # Fall back to u.gg scraping
        # Normalize player name for URL
        # Handle Riot ID format (gameName#tagLine) -> gameName-tagLine
        original_name = player_name
        if '#' in player_name:
            game_name, tag_line = player_name.split('#', 1)
            player_normalized = f"{game_name}-{tag_line}".replace(' ', '-').lower()
        else:
            player_normalized = player_name.replace('#', '-').replace(' ', '-').lower()
        
        # Normalize champion name - Riot API might return "JarvanIV" but u.gg uses "Jarvan IV"
        champion_normalized = self._normalize_champion_name_for_ugg(champion)
        
        # Build u.gg profile URL
        url = f"https://u.gg/lol/profile/{region}/{player_normalized}/champion-stats"
        
        if debug:
            print(f"  [DEBUG] Player: {original_name} -> {player_normalized}")
            print(f"  [DEBUG] Champion: {champion} -> {champion_normalized}")
            print(f"  [DEBUG] URL: {url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            if debug:
                print(f"  [DEBUG] Response status: {response.status_code}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Strategy 1: Look for JSON data in script tags (most reliable)
            scripts = soup.find_all('script', type='application/json')
            if debug:
                print(f"  [DEBUG] Found {len(scripts)} JSON script tags")
            
            for idx, script in enumerate(scripts):
                try:
                    import json
                    script_content = script.string
                    if not script_content:
                        continue
                    
                    data = json.loads(script_content)
                    
                    if debug:
                        # Print a sample of the JSON structure to help debug
                        if idx == 0:
                            import json as json_module
                            json_str = json_module.dumps(data, indent=2)[:2000]  # First 2000 chars
                            print(f"  [DEBUG] First JSON structure sample:\n{json_str}...")
                    
                    champ_stats = self._extract_champion_stats_from_json(data, champion_normalized, debug=debug)
                    if champ_stats:
                        games = champ_stats.get('games', 0)
                        winrate = champ_stats.get('winrate', 0.5)
                        if debug:
                            print(f"  [DEBUG] Found in JSON: Games={games}, WR={winrate:.2%}")
                        if games >= min_games:
                            return {
                                'winrate': winrate,
                                'games_played': games
                            }
                        else:
                            return {
                                'winrate': 0.5,  # Use default if not enough games
                                'games_played': games
                            }
                except (json.JSONDecodeError, AttributeError) as e:
                    if debug:
                        print(f"  [DEBUG] JSON parse error: {e}")
                    continue
                except Exception as e:
                    if debug:
                        print(f"  [DEBUG] Unexpected error parsing JSON: {e}")
                    continue
            
            # Strategy 2: Parse HTML table for champion stats
            # Look for tables with champion data
            tables = soup.find_all('table')
            if debug:
                print(f"  [DEBUG] Found {len(tables)} tables")
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                    
                    # Get all text from the row
                    row_text = ' '.join([cell.get_text(strip=True) for cell in cells])
                    
                    # Try both original and normalized champion name
                    if (champion.lower() in row_text.lower() or 
                        champion_normalized.lower() in row_text.lower()):
                        
                        if debug:
                            print(f"  [DEBUG] Found champion in row: {row_text[:200]}")
                        
                        # Format 1: "50% / 9W 9L" or "73% / 8W 3L"
                        # Pattern: percentage followed by slash and W-L record
                        format1_pattern = r'(\d+\.?\d*)\s*%\s*/\s*(\d+)\s*[Ww]\s*(\d+)\s*[Ll]'
                        format1_match = re.search(format1_pattern, row_text)
                        if format1_match:
                            wr_pct = float(format1_match.group(1))
                            wins = int(format1_match.group(2))
                            losses = int(format1_match.group(3))
                            games = wins + losses
                            
                            if debug:
                                print(f"  [DEBUG] Format 1 match: {wr_pct}% / {wins}W {losses}L = {games} games")
                            
                            if games > 0:
                                winrate = wr_pct / 100.0
                                if games >= min_games:
                                    return {'winrate': winrate, 'games_played': games}
                                else:
                                    return {'winrate': 0.5, 'games_played': games}
                        
                        # Format 2: Separate "18W" and "17L" with win rate percentage nearby
                        # Look for pattern: "18W" and "17L" in the same row, plus a percentage
                        format2_w_pattern = r'(\d+)\s*[Ww](?!\s*\d+\s*[Ll])'  # Wins not immediately followed by losses
                        format2_l_pattern = r'(\d+)\s*[Ll]'
                        format2_wr_pattern = r'(\d+\.?\d*)\s*%'
                        
                        wins_match = re.search(format2_w_pattern, row_text)
                        losses_match = re.search(format2_l_pattern, row_text)
                        wr_match = re.search(format2_wr_pattern, row_text)
                        
                        if wins_match and losses_match:
                            wins = int(wins_match.group(1))
                            losses = int(losses_match.group(1))
                            games = wins + losses
                            
                            # Use winrate from percentage if available, otherwise calculate
                            if wr_match:
                                winrate = float(wr_match.group(1)) / 100.0
                            else:
                                winrate = wins / games if games > 0 else 0.5
                            
                            if debug:
                                print(f"  [DEBUG] Format 2 match: {wins}W {losses}L = {games} games, WR={winrate:.2%}")
                            
                            if games > 0:
                                if games >= min_games:
                                    return {'winrate': winrate, 'games_played': games}
                                else:
                                    return {'winrate': 0.5, 'games_played': games}
                        
                        # Format 3: Simple "15W 10L" pattern (no percentage)
                        wl_pattern = r'(\d+)\s*[Ww]\s*(\d+)\s*[Ll]'
                        wl_match = re.search(wl_pattern, row_text)
                        
                        if wl_match:
                            wins = int(wl_match.group(1))
                            losses = int(wl_match.group(2))
                            games = wins + losses
                            
                            if debug:
                                print(f"  [DEBUG] Format 3 match: {wins}W {losses}L = {games} games")
                            
                            if games > 0:
                                winrate = wins / games
                                if games >= min_games:
                                    return {'winrate': winrate, 'games_played': games}
                                else:
                                    return {'winrate': 0.5, 'games_played': games}
                        
                        # Format 4: Look for games count and winrate percentage separately
                        games_match = re.search(r'(\d+)\s*games?', row_text, re.I)
                        wr_match = re.search(r'(\d+\.?\d*)\s*%', row_text)
                        
                        if games_match and wr_match:
                            games = int(games_match.group(1))
                            winrate = float(wr_match.group(1)) / 100.0
                            
                            if debug:
                                print(f"  [DEBUG] Format 4 match: {games} games, WR={winrate:.2%}")
                            
                            if games >= min_games:
                                return {'winrate': winrate, 'games_played': games}
                            else:
                                return {'winrate': 0.5, 'games_played': games}
            
            # Strategy 3: Look for champion name in any element and find nearby stats
            # Try both original and normalized names
            champ_pattern = f"({'|'.join([re.escape(champion), re.escape(champion_normalized)])})"
            champ_elements = soup.find_all(text=re.compile(champ_pattern, re.I))
            if debug:
                print(f"  [DEBUG] Found {len(champ_elements)} elements matching champion name")
            
            for elem in champ_elements:
                # Navigate up to find parent container
                parent = elem.parent
                for _ in range(5):  # Check up to 5 levels up
                    if parent is None:
                        break
                    
                    # Get all text in the container
                    container_text = parent.get_text()
                    
                    if debug and _ == 0:
                        print(f"  [DEBUG] Checking container text: {container_text[:200]}")
                    
                    # Format 1: "50% / 9W 9L"
                    format1_pattern = r'(\d+\.?\d*)\s*%\s*/\s*(\d+)\s*[Ww]\s*(\d+)\s*[Ll]'
                    format1_match = re.search(format1_pattern, container_text)
                    if format1_match:
                        wr_pct = float(format1_match.group(1))
                        wins = int(format1_match.group(2))
                        losses = int(format1_match.group(3))
                        games = wins + losses
                        
                        if debug:
                            print(f"  [DEBUG] Format 1 match in container: {wr_pct}% / {wins}W {losses}L")
                        
                        if games > 0:
                            winrate = wr_pct / 100.0
                            if games >= min_games:
                                return {'winrate': winrate, 'games_played': games}
                            else:
                                return {'winrate': 0.5, 'games_played': games}
                    
                    # Format 2: Separate "18W" and "17L" with percentage
                    wins_match = re.search(r'(\d+)\s*[Ww]', container_text)
                    losses_match = re.search(r'(\d+)\s*[Ll]', container_text)
                    wr_match = re.search(r'(\d+\.?\d*)\s*%', container_text)
                    
                    if wins_match and losses_match:
                        wins = int(wins_match.group(1))
                        losses = int(losses_match.group(1))
                        games = wins + losses
                        
                        if wr_match:
                            winrate = float(wr_match.group(1)) / 100.0
                        else:
                            winrate = wins / games if games > 0 else 0.5
                        
                        if debug:
                            print(f"  [DEBUG] Format 2 match in container: {wins}W {losses}L, WR={winrate:.2%}")
                        
                        if games > 0:
                            if games >= min_games:
                                return {'winrate': winrate, 'games_played': games}
                            else:
                                return {'winrate': 0.5, 'games_played': games}
                    
                    # Format 3: Simple "15W 10L"
                    wl_pattern = r'(\d+)\s*[Ww]\s*(\d+)\s*[Ll]'
                    wl_match = re.search(wl_pattern, container_text)
                    if wl_match:
                        wins = int(wl_match.group(1))
                        losses = int(wl_match.group(2))
                        games = wins + losses
                        
                        if debug:
                            print(f"  [DEBUG] Format 3 match in container: {wins}W {losses}L")
                        
                        if games > 0:
                            winrate = wins / games
                            if games >= min_games:
                                return {'winrate': winrate, 'games_played': games}
                            else:
                                return {'winrate': 0.5, 'games_played': games}
                    
                    # Format 4: Games count and winrate percentage
                    games_match = re.search(r'(\d+)\s*games?', container_text, re.I)
                    wr_match = re.search(r'(\d+\.?\d*)\s*%', container_text)
                    
                    if games_match and wr_match:
                        games = int(games_match.group(1))
                        winrate = float(wr_match.group(1)) / 100.0
                        
                        if debug:
                            print(f"  [DEBUG] Format 4 match in container: {games} games, WR={winrate:.2%}")
                        
                        if games >= min_games:
                            return {'winrate': winrate, 'games_played': games}
                        else:
                            return {'winrate': 0.5, 'games_played': games}
                    
                    parent = parent.parent if hasattr(parent, 'parent') else None
            
            # u.gg didn't find the champion, try OP.GG
            ugg_result = None
            
        except requests.RequestException:
            if debug:
                print(f"  [DEBUG] u.gg request failed, trying OP.GG...")
            ugg_result = None
        except (ValueError, AttributeError) as e:
            if debug:
                print(f"  [DEBUG] u.gg parsing failed: {e}, trying OP.GG...")
            ugg_result = None
        
        # Fall back to OP.GG scraping if u.gg didn't return a result
        if ugg_result is None:
            try:
                if debug:
                    print(f"  [DEBUG] Attempting OP.GG fallback...")
                result = self._get_player_champion_winrate_from_opgg(
                    player_name, champion, region, min_games, debug
                )
                if result:
                    if debug:
                        print(f"  [DEBUG] ✓ Successfully retrieved from OP.GG")
                    return result
            except Exception as e:
                if debug:
                    print(f"  [DEBUG] OP.GG scraping also failed: {e}")
        
        return None
    
    def _get_player_champion_winrate_from_opgg(self, player_name: str, champion: str, 
                                                region: str = 'na1', min_games: int = 10, 
                                                debug: bool = False) -> Optional[Dict]:
        """
        Get a player's win rate on a specific champion from OP.GG.
        Uses caching to avoid redundant API calls for the same player-champion combination.
        
        Args:
            player_name: Summoner name or Riot ID (e.g., 'dtboss-2003' or 'dtboss#2003')
            champion: Champion name
            region: Region code (default 'na1')
            min_games: Minimum games required (default 10)
            debug: If True, print debug information
            
        Returns:
            Dictionary with 'winrate' (float), 'games_played' (int), or None if not found
        """
        # Check cache first - use normalized names for cache key
        cache_key = (player_name.lower().strip(), champion.lower().strip(), region.lower())
        if cache_key in self.player_comfort_cache:
            cached_result = self.player_comfort_cache[cache_key]
            if debug:
                print(f"  [DEBUG] [OP.GG] Cache HIT for {player_name} on {champion}")
            return cached_result
        
        # Normalize player name for URL
        original_name = player_name
        if '#' in player_name:
            game_name, tag_line = player_name.split('#', 1)
            player_normalized = f"{game_name}-{tag_line}".replace(' ', '-').lower()
        else:
            player_normalized = player_name.replace('#', '-').replace(' ', '-').lower()
        
        # OP.GG uses different region codes
        opgg_region_map = {
            'na1': 'na', 'euw1': 'euw', 'eun1': 'eune',
            'kr': 'kr', 'jp1': 'jp', 'br1': 'br',
            'la1': 'lan', 'la2': 'las', 'oc1': 'oce',
            'tr1': 'tr', 'ru': 'ru'
        }
        opgg_region = opgg_region_map.get(region.lower(), 'na')
        
        # Normalize champion name for matching
        champion_normalized = self._normalize_champion_name_for_ugg(champion)
        
        # Build OP.GG profile URL
        url = f"https://op.gg/lol/summoners/{opgg_region}/{player_normalized}/champions"
        
        if debug:
            print(f"  [DEBUG] [OP.GG] Player: {original_name} -> {player_normalized}")
            print(f"  [DEBUG] [OP.GG] Champion: {champion} -> {champion_normalized}")
            print(f"  [DEBUG] [OP.GG] URL: {url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            if debug:
                print(f"  [DEBUG] [OP.GG] Response status: {response.status_code}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Print page structure and save HTML to file
            if debug:
                print(f"  [DEBUG] [OP.GG] Page title: {soup.title.string if soup.title else 'No title'}")
                
                # Save full HTML to file for inspection
                try:
                    html_filename = f"opgg_debug_{player_normalized}_{champion_normalized}.html"
                    with open(html_filename, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    print(f"  [DEBUG] [OP.GG] Saved full HTML to: {html_filename}")
                except Exception as e:
                    print(f"  [DEBUG] [OP.GG] Could not save HTML file: {e}")
                
                # Check for __NEXT_DATA__ script tag
                next_data_tags = soup.find_all('script', id='__NEXT_DATA__')
                print(f"  [DEBUG] [OP.GG] Found {len(next_data_tags)} __NEXT_DATA__ script tag(s)")
                
                # Check for other script tags
                all_scripts = soup.find_all('script')
                print(f"  [DEBUG] [OP.GG] Total script tags: {len(all_scripts)}")
                for i, script in enumerate(all_scripts[:10]):  # First 10 scripts
                    script_id = script.get('id', 'no-id')
                    script_type = script.get('type', 'no-type')
                    script_len = len(script.string) if script.string else 0
                    print(f"  [DEBUG] [OP.GG]   Script {i+1}: id={script_id}, type={script_type}, length={script_len}")
                
                # Print a sample of the HTML structure
                print(f"  [DEBUG] [OP.GG] Sample HTML (first 2000 chars):")
                print(f"  {str(soup)[:2000]}")
                print(f"  [DEBUG] [OP.GG] ...")
            
            # Strategy 1: Try to extract JSON from __NEXT_DATA__ (most reliable)
            script_tag = soup.select_one("script#__NEXT_DATA__")
            if script_tag:
                if debug:
                    print(f"  [DEBUG] [OP.GG] Found __NEXT_DATA__ script tag")
                
                try:
                    import json
                    raw_json = script_tag.string
                    if raw_json:
                        data = json.loads(raw_json)
                        
                        # Navigate to champions list
                        # Path: data["props"]["pageProps"]["champions"]["champions"]
                        champ_list = (data
                                    .get("props", {})
                                    .get("pageProps", {})
                                    .get("champions", {})
                                    .get("champions", []))
                        
                        if debug:
                            print(f"  [DEBUG] [OP.GG] Found {len(champ_list)} champions in JSON data")
                        
                        # Normalize champion name for matching
                        champion_lower = champion.lower().strip()
                        champion_normalized_lower = champion_normalized.lower().strip()
                        
                        for champ in champ_list:
                            champ_name = champ.get("championName") or champ.get("champion") or champ.get("name")
                            if not champ_name:
                                continue
                            
                            champ_name_lower = str(champ_name).lower().strip()
                            
                            # Match champion name (exact match or normalized match)
                            if (champ_name_lower == champion_lower or 
                                champ_name_lower == champion_normalized_lower):
                                
                                if debug:
                                    print(f"  [DEBUG] [OP.GG] Found champion in JSON: {champ_name}")
                                
                                # Extract winrate and games
                                win_rate = champ.get("winRate")  # might be in decimal e.g. 0.5123
                                wins = champ.get("wins") or champ.get("win") or 0
                                losses = champ.get("losses") or champ.get("loss") or 0
                                games = champ.get("games") or champ.get("gamesPlayed") or champ.get("totalGames") or 0
                                
                                # Calculate games from wins/losses if not provided
                                if games == 0 and (wins > 0 or losses > 0):
                                    games = wins + losses
                                
                                # Use winrate from data if available, otherwise calculate
                                if win_rate is not None:
                                    # If winrate is > 1, assume it's a percentage (e.g., 51.23)
                                    if win_rate > 1:
                                        winrate = win_rate / 100.0
                                    else:
                                        winrate = win_rate
                                elif games > 0 and isinstance(wins, (int, float)) and isinstance(losses, (int, float)):
                                    winrate = wins / games
                                else:
                                    winrate = 0.5
                                
                                if debug:
                                    print(f"  [DEBUG] [OP.GG] Extracted from JSON: Games={games}, Wins={wins}, Losses={losses}, WR={winrate:.2%}")
                                
                                if games > 0:
                                    if games >= min_games:
                                        result = {'winrate': winrate, 'games_played': int(games)}
                                        self.player_comfort_cache[cache_key] = result
                                        return result
                                    else:
                                        result = {'winrate': 0.5, 'games_played': int(games)}
                                        self.player_comfort_cache[cache_key] = result
                                        return result
                        
                        if debug:
                            print(f"  [DEBUG] [OP.GG] Champion not found in JSON data")
                
                except (json.JSONDecodeError, KeyError, AttributeError) as e:
                    if debug:
                        print(f"  [DEBUG] [OP.GG] JSON parsing error: {e}, falling back to HTML parsing")
                    # Fall through to HTML parsing
            
            # Strategy 2: Fall back to HTML table parsing if JSON extraction fails
            if debug:
                print(f"  [DEBUG] [OP.GG] Attempting HTML table parsing...")
            
            tables = soup.find_all('table')
            if debug:
                print(f"  [DEBUG] [OP.GG] Found {len(tables)} tables")
                
                # Print table structure for debugging
                for i, table in enumerate(tables):
                    print(f"  [DEBUG] [OP.GG] Table {i+1}:")
                    rows = table.find_all('tr')
                    print(f"    Rows: {len(rows)}")
                    
                    # Print first few rows
                    for j, row in enumerate(rows[:3]):
                        cells = row.find_all(['td', 'th'])
                        row_text = ' '.join([cell.get_text(strip=True)[:50] for cell in cells[:5]])
                        print(f"    Row {j+1}: {len(cells)} cells - {row_text[:200]}")
                    
                    # Look for champion names in the table
                    all_text = table.get_text()
                    if champion.lower() in all_text.lower() or champion_normalized.lower() in all_text.lower():
                        print(f"    ✓ Champion name found in table text!")
                    else:
                        print(f"    ✗ Champion name NOT found in table text")
                    
                    # Print table HTML structure (first 1000 chars)
                    print(f"    HTML sample: {str(table)[:1000]}")
                    print(f"    ...")
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    # Skip header row
                    if row.find('thead') or not row.find('td'):
                        continue
                    
                    # Get all text from the row for easier matching
                    row_text = row.get_text()
                    
                    # Check if champion name appears in the row text
                    # Format might be: "1 Yasuo 18W17L51%" or similar
                    champion_lower = champion.lower().strip()
                    champion_normalized_lower = champion_normalized.lower().strip()
                    row_text_lower = row_text.lower()
                    
                    # Match champion name in row text (more flexible matching)
                    champ_found = False
                    if (champion_lower in row_text_lower or 
                        champion_normalized_lower in row_text_lower):
                        champ_found = True
                        
                        if debug:
                            print(f"  [DEBUG] [OP.GG] Found champion '{champion}' in row text: {row_text[:200]}")
                    
                    # Also try to find champion name in specific cells
                    if not champ_found:
                        # Find champion name in the row
                        # OP.GG structure: <td class="champion"><strong>ChampionName</strong></td>
                        champ_cell = row.find('td', class_='champion')
                        if not champ_cell:
                            # Try finding by strong tag or img alt
                            cells = row.find_all('td')
                            for cell in cells:
                                strong_tag = cell.find('strong')
                                if strong_tag:
                                    champ_cell = strong_tag
                                    break
                                img_tag = cell.find('img')
                                if img_tag and img_tag.get('alt'):
                                    champ_cell = img_tag
                                    break
                        
                        if champ_cell:
                            # Get champion name from text or alt attribute
                            if champ_cell.name == 'img':
                                champ_name_text = champ_cell.get('alt', '')
                            else:
                                # Get text from strong tag or the cell itself
                                strong = champ_cell.find('strong')
                                if strong:
                                    champ_name_text = strong.get_text(strip=True)
                                else:
                                    champ_name_text = champ_cell.get_text(strip=True)
                            
                            if champ_name_text:
                                champ_name_lower = champ_name_text.lower().strip()
                                if (champ_name_lower == champion_lower or 
                                    champ_name_lower == champion_normalized_lower):
                                    champ_found = True
                                    if debug:
                                        print(f"  [DEBUG] [OP.GG] Found champion in HTML cell: {champ_name_text}")
                    
                    if champ_found:
                        # Extract win/loss data from the row
                        # OP.GG format: "18W17L51%" or "18W 17L 51%" or similar
                        # Pattern: Look for "18W" and "17L" followed by percentage
                        # Handle formats like: "18W17L51%" or "18W 17L 51%" or "18W17L 51%"
                        
                        # Pattern 1: "18W17L51%" (no spaces)
                        pattern1 = r'(\d+)\s*[Ww]\s*(\d+)\s*[Ll]\s*(\d+\.?\d*)\s*%'
                        match1 = re.search(pattern1, row_text)
                        
                        # Pattern 2: "18W17L" followed by "51%" somewhere
                        pattern2_w = r'(\d+)\s*[Ww](?!\s*\d+\s*[Ll])'
                        pattern2_l = r'(\d+)\s*[Ll]'
                        pattern2_wr = r'(\d+\.?\d*)\s*%'
                        
                        wins_match = re.search(pattern2_w, row_text)
                        losses_match = re.search(pattern2_l, row_text)
                        wr_match = re.search(pattern2_wr, row_text)
                        
                        if match1:
                            # Format: "18W17L51%"
                            wins = int(match1.group(1))
                            losses = int(match1.group(2))
                            winrate_pct = float(match1.group(3))
                            games = wins + losses
                            winrate = winrate_pct / 100.0
                            
                            if debug:
                                print(f"  [DEBUG] [OP.GG] Extracted (pattern1): {wins}W {losses}L = {games} games, WR={winrate:.2%}")
                            
                            if games > 0:
                                if games >= min_games:
                                    result = {'winrate': winrate, 'games_played': games}
                                    self.player_comfort_cache[cache_key] = result
                                    return result
                                else:
                                    result = {'winrate': 0.5, 'games_played': games}
                                    self.player_comfort_cache[cache_key] = result
                                    return result
                        
                        elif wins_match and losses_match:
                            # Format: "18W" and "17L" separate, with optional percentage
                            wins = int(wins_match.group(1))
                            losses = int(losses_match.group(1))
                            games = wins + losses
                            
                            # Use winrate from percentage if available, otherwise calculate
                            if wr_match:
                                winrate = float(wr_match.group(1)) / 100.0
                            else:
                                winrate = wins / games if games > 0 else 0.5
                            
                            if debug:
                                print(f"  [DEBUG] [OP.GG] Extracted (pattern2): {wins}W {losses}L = {games} games, WR={winrate:.2%}")
                            
                            if games > 0:
                                if games >= min_games:
                                    return {'winrate': winrate, 'games_played': games}
                                else:
                                    return {'winrate': 0.5, 'games_played': games}
            
            if debug:
                print(f"  [DEBUG] [OP.GG] Champion not found")
            # Cache None to avoid retrying failed lookups
            self.player_comfort_cache[cache_key] = None
            return None
            
        except requests.RequestException as e:
            if debug:
                print(f"  [DEBUG] [OP.GG] Request failed: {e}")
            # Cache None to avoid retrying failed requests
            self.player_comfort_cache[cache_key] = None
            return None
        except (ValueError, AttributeError) as e:
            if debug:
                print(f"  [DEBUG] [OP.GG] Parsing failed: {e}")
            # Cache None to avoid retrying failed parsing
            self.player_comfort_cache[cache_key] = None
            return None
    
    def _extract_champion_stats_from_json(self, data, champion_name: str, debug: bool = False) -> Optional[Dict]:
        """Extract champion statistics from JSON data structure."""
        # Normalize champion name for comparison - try multiple variations
        champ_normalized = champion_name.lower().strip()
        champ_normalized_ugg = self._normalize_champion_name_for_ugg(champion_name).lower().strip()
        champ_variations = [
            champ_normalized,
            champion_name.strip().lower(),
            champ_normalized_ugg,
            self._normalize_champion_name_for_ugg(champion_name).strip().lower()
        ]
        
        if debug:
            print(f"    [DEBUG] Looking for champion: {champion_name}")
            print(f"    [DEBUG] Variations: {champ_variations}")
        
        if isinstance(data, dict):
            # Look for champion stats in various possible keys
            # u.gg might use different key names
            possible_keys = ['champions', 'championStats', 'champion_stats', 'championStatsData', 
                           'stats', 'championData', 'playerChampionStats', 'championList']
            
            champions = None
            for key in possible_keys:
                if key in data:
                    champions = data[key]
                    if debug:
                        print(f"    [DEBUG] Found champions in key: {key}")
                    break
            
            # Also check if data itself is a list of champions
            if champions is None and isinstance(data, dict):
                # Check if any value is a list that might contain champions
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Check if first item looks like a champion object
                        if isinstance(value[0], dict):
                            first_item = value[0]
                            if any(k in first_item for k in ['championName', 'champion', 'name', 'champ', 'championId']):
                                champions = value
                                if debug:
                                    print(f"    [DEBUG] Found champions list in key: {key}")
                                break
            
            # Process champions list if found
            if isinstance(champions, list):
                for champ in champions:
                    if isinstance(champ, dict):
                        # Try multiple possible name fields
                        champ_name_raw = (champ.get('championName') or 
                                        champ.get('champion') or 
                                        champ.get('name') or
                                        champ.get('champ') or
                                        champ.get('championId') or
                                        champ.get('champion_id', ''))
                        
                        if not champ_name_raw:
                            continue
                            
                        champ_name = str(champ_name_raw).lower().strip()
                        
                        if debug:
                            print(f"    [DEBUG] Checking champion: {champ_name_raw} -> {champ_name}")
                        
                        # Match champion name (exact match only to avoid false positives) - try all variations
                        matched = False
                        for champ_var in champ_variations:
                            # Use exact equality only - no substring matching to prevent false matches
                            if (champ_var == champ_name or 
                                str(champ_name_raw).lower().strip() == champ_var):
                                matched = True
                                if debug:
                                    print(f"    [DEBUG] ✓ Matched! Using variation: {champ_var}")
                                break
                        
                        if matched:
                            
                            # Try to get games and wins from various possible fields
                            games = (champ.get('gamesPlayed') or 
                                   champ.get('games') or 
                                   champ.get('count') or
                                   champ.get('totalGames') or
                                   champ.get('matchCount') or
                                   champ.get('numGames') or
                                   0)
                            
                            wins = (champ.get('wins') or 
                                  champ.get('win') or
                                  champ.get('winsCount') or
                                  champ.get('victories') or
                                  0)
                            
                            # If we have wins and games, calculate winrate
                            if isinstance(games, (int, float)) and games > 0:
                                if isinstance(wins, (int, float)):
                                    winrate = wins / games
                                else:
                                    # Try to calculate from winrate percentage
                                    wr_pct = champ.get('winrate') or champ.get('winRate') or champ.get('win_rate')
                                    if wr_pct:
                                        if isinstance(wr_pct, str):
                                            wr_pct = float(wr_pct.replace('%', ''))
                                        winrate = wr_pct / 100.0 if wr_pct > 1 else wr_pct
                                    else:
                                        winrate = 0.5
                                
                                if debug:
                                    print(f"    [DEBUG] Extracted stats: Games={games}, Wins={wins}, WR={winrate:.2%}")
                                
                                return {
                                    'winrate': winrate,
                                    'games': int(games),
                                    'wins': int(wins) if isinstance(wins, (int, float)) else 0
                                }
                            
                            # If we only have winrate percentage
                            wr_pct = champ.get('winrate') or champ.get('winRate') or champ.get('win_rate')
                            if wr_pct and isinstance(games, (int, float)) and games > 0:
                                if isinstance(wr_pct, str):
                                    wr_pct = float(wr_pct.replace('%', ''))
                                winrate = wr_pct / 100.0 if wr_pct > 1 else wr_pct
                                if debug:
                                    print(f"    [DEBUG] Extracted from WR%: Games={games}, WR={winrate:.2%}")
                                return {
                                    'winrate': winrate,
                                    'games': int(games),
                                    'wins': 0
                                }
            
            # Recursively search nested structures
            for value in data.values():
                result = self._extract_champion_stats_from_json(value, champion_name, debug=debug)
                if result:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._extract_champion_stats_from_json(item, champion_name, debug=debug)
                if result:
                    return result
        
        return None
    
    def calculate_player_comfort_score(self, blue_players: List[str], red_players: List[str],
                                       blue_team: List[str], red_team: List[str],
                                       region: str = 'na1', w2: float = 0.35,
                                       return_details: bool = False,
                                       use_database: bool = None) -> float:
        """
        Calculate player comfort score: w2 * ∑(Player WR on champ - Enemy WR on champ) for all 5 roles.
        
        Formula: Sum of (Blue player's winrate on their champion - Red player's winrate on their champion)
        across all 5 roles. If a player has < 10 games on their champion, use 0.5 as their winrate.
        Then multiply by w2 and clamp to [-0.15, 0.15].
        
        Args:
            blue_players: List of 5 blue side player names [top, jungle, mid, adc, support]
            red_players: List of 5 red side player names [top, jungle, mid, adc, support]
            blue_team: List of 5 blue side champions [top, jungle, mid, adc, support]
            red_team: List of 5 red side champions [top, jungle, mid, adc, support]
            region: Region code (default 'na1')
            w2: Weight factor (default 0.35)
            return_details: If True, returns dict with score and breakdown details
            use_database: If True, use database (worlds2025.db). If False, use OP.GG scraping.
                          If None (default), use database when available, otherwise fall back to OP.GG.
            
        Returns:
            Player comfort score clamped to [-0.15, 0.15], or dict with details if return_details=True
        """
        # Default to database if available, otherwise use OP.GG
        if use_database is None:
            use_database = HAS_PLAYER_COMFORT_DB
        if len(blue_players) != 5 or len(red_players) != 5:
            raise ValueError("Both teams must have exactly 5 players")
        if len(blue_team) != 5 or len(red_team) != 5:
            raise ValueError("Both teams must have exactly 5 champions")
        
        role_order = ['top', 'jungle', 'mid', 'adc', 'support']
        comfort_differences = []
        comfort_details = []
        
        for i, role in enumerate(role_order):
            blue_player = blue_players[i]
            red_player = red_players[i]
            blue_champ = blue_team[i]
            red_champ = red_team[i]

            # If either side hasn't picked this role yet, treat comfort as neutral and skip lookups
            if blue_champ is None or red_champ is None:
                comfort_differences.append(0.0)
                comfort_details.append({
                    'role': role,
                    'blue_player': blue_player,
                    'red_player': red_player,
                    'blue_champ': blue_champ,
                    'red_champ': red_champ,
                    'blue_wr': None,
                    'red_wr': None,
                    'blue_games': 0,
                    'red_games': 0,
                    'difference': 0.0,
                })
                continue
            
            # Get player winrates - use database if available and requested, otherwise use OP.GG scraping
            # Cache key for player comfort: (player_name, champion_name)
            blue_cache_key = (blue_player.lower().strip(), blue_champ.lower().strip())
            red_cache_key = (red_player.lower().strip(), red_champ.lower().strip())
            
            # Check cache first
            if blue_cache_key in self.player_comfort_cache:
                blue_stats = self.player_comfort_cache[blue_cache_key]
            else:
                if use_database and HAS_PLAYER_COMFORT_DB:
                    # Load from database (fast, from worlds2025.db)
                    blue_comfort = get_player_comfort(blue_player, blue_champ, min_games=10)
                    blue_stats = {
                        'winrate': blue_comfort['winrate'],
                        'games_played': blue_comfort['games']
                    } if blue_comfort['found'] else None
                else:
                    # Fall back to OP.GG scraping (slower, but works for any player)
                    blue_stats = self._get_player_champion_winrate_from_opgg(blue_player, blue_champ, region, min_games=10, debug=False)
                # Cache the result (even if None)
                self.player_comfort_cache[blue_cache_key] = blue_stats
            
            if red_cache_key in self.player_comfort_cache:
                red_stats = self.player_comfort_cache[red_cache_key]
            else:
                if use_database and HAS_PLAYER_COMFORT_DB:
                    # Load from database (fast, from worlds2025.db)
                    red_comfort = get_player_comfort(red_player, red_champ, min_games=10)
                    red_stats = {
                        'winrate': red_comfort['winrate'],
                        'games_played': red_comfort['games']
                    } if red_comfort['found'] else None
                else:
                    # Fall back to OP.GG scraping (slower, but works for any player)
                    red_stats = self._get_player_champion_winrate_from_opgg(red_player, red_champ, region, min_games=10, debug=False)
                # Cache the result (even if None)
                self.player_comfort_cache[red_cache_key] = red_stats
            
            # Use winrates (0.5 if not enough games or not found)
            blue_wr = blue_stats['winrate'] if blue_stats else 0.5
            red_wr = red_stats['winrate'] if red_stats else 0.5
            
            blue_games = blue_stats['games_played'] if blue_stats else 0
            red_games = red_stats['games_played'] if red_stats else 0
            
            # Calculate difference: Blue player WR - Red player WR
            diff = blue_wr - red_wr
            comfort_differences.append(diff)
            
            comfort_details.append({
                'role': role,
                'blue_player': blue_player,
                'blue_champ': blue_champ,
                'blue_wr': blue_wr,
                'blue_games': blue_games,
                'red_player': red_player,
                'red_champ': red_champ,
                'red_wr': red_wr,
                'red_games': red_games,
                'difference': diff
            })
        
        # Sum all differences
        sum_differences = sum(comfort_differences)
        
        # Apply weight: w2 * ∑(Player WR - Enemy WR)
        raw_score = w2 * sum_differences
        
        # Clamp to [-0.15, 0.15]
        clamped_score = max(-0.15, min(0.15, raw_score))
        
        if return_details:
            return {
                'score': clamped_score,
                'raw_score': raw_score,
                'sum_differences': sum_differences,
                'w2': w2,
                'comfort_details': comfort_details
            }
        
        return clamped_score
    
    def get_puuid_from_riot_id(self, game_name: str, tag_line: str, region: str = 'americas') -> str:
        """
        Get PUUID from Riot ID (gameName#tagLine).
        
        Args:
            game_name: Player's game name (e.g., 'dtboss')
            tag_line: Player's tag line (e.g., '2003')
            region: API region (default 'americas' for NA, 'europe' for EU, 'asia' for KR/JP/etc)
            
        Returns:
            PUUID string
        """
        if not self.riot_api_key:
            raise ValueError(
                "Riot API key required. Set it when initializing CompositionScorer: "
                "CompositionScorer(riot_api_key='YOUR_KEY')"
            )
        
        url = f"https://{region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        headers = {'X-Riot-Token': self.riot_api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('puuid', '')
        except requests.RequestException as e:
            raise ValueError(f"Failed to get PUUID from Riot ID: {e}")
    
    def get_match_history_by_puuid(self, puuid: str, region: str = 'americas', 
                                   start: int = 0, count: int = 20, 
                                   queue_id: int = None) -> List[str]:
        """
        Get match history (list of match IDs) from PUUID.
        
        Args:
            puuid: Player's PUUID
            region: API region (default 'americas')
            start: Start index (default 0)
            count: Number of matches to retrieve (default 20, max 100)
            queue_id: Optional queue filter (e.g., 420 for Ranked Solo/Duo)
            
        Returns:
            List of match IDs
        """
        if not self.riot_api_key:
            raise ValueError(
                "Riot API key required. Set it when initializing CompositionScorer: "
                "CompositionScorer(riot_api_key='YOUR_KEY')"
            )
        
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        headers = {'X-Riot-Token': self.riot_api_key}
        params = {'start': start, 'count': count}
        
        if queue_id:
            params['queue'] = queue_id
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ValueError(f"Failed to get match history: {e}")
    
    def get_match_history_by_riot_id(self, game_name: str, tag_line: str, 
                                     region: str = 'americas', start: int = 0, 
                                     count: int = 20, queue_id: int = None) -> List[str]:
        """
        Get match history from Riot ID (gameName#tagLine).
        Convenience function that combines get_puuid_from_riot_id and get_match_history_by_puuid.
        
        Args:
            game_name: Player's game name (e.g., 'dtboss')
            tag_line: Player's tag line (e.g., '2003')
            region: API region (default 'americas')
            start: Start index (default 0)
            count: Number of matches to retrieve (default 20, max 100)
            queue_id: Optional queue filter (e.g., 420 for Ranked Solo/Duo)
            
        Returns:
            List of match IDs
        """
        puuid = self.get_puuid_from_riot_id(game_name, tag_line, region)
        return self.get_match_history_by_puuid(puuid, region, start, count, queue_id)
    
    def fetch_match_by_riot_id(self, match_id: str, region: str = 'americas') -> Dict:
        """
        Fetch complete match data using Riot Games API match ID.
        
        Args:
            match_id: Riot Games match ID (e.g., 'NA1_1234567890')
            region: API region (default 'americas' for NA, 'europe' for EU, 'asia' for KR/JP/etc)
            
        Returns:
            Dictionary with blue_team, red_team, blue_players, red_players, and other match info
        """
        if not self.riot_api_key:
            raise ValueError(
                "Riot API key required. Set it when initializing CompositionScorer: "
                "CompositionScorer(riot_api_key='YOUR_KEY')"
            )
        
        # Riot API endpoint
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        headers = {
            'X-Riot-Token': self.riot_api_key
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract match data from Riot API response
            return self._parse_riot_match_data(data)
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch match data from Riot API: {e}")
    
    def _parse_riot_match_data(self, data: Dict) -> Dict:
        """Parse match data from Riot Games API response."""
        info = data.get('info', {})
        participants = info.get('participants', [])
        
        if len(participants) != 10:
            raise ValueError(f"Expected 10 participants, got {len(participants)}")
        
        blue_team = []
        red_team = []
        blue_players = []
        red_players = []
        
        # Riot API: teamId 100 = blue side, 200 = red side
        for participant in participants:
            team_id = participant.get('teamId', 0)
            champion = participant.get('championName', '')
            summoner_name = participant.get('summonerName', '')
            riot_id = participant.get('riotIdGameName', '')
            riot_tag = participant.get('riotIdTagline', '')
            
            # Use Riot ID if available, otherwise summoner name
            player_name = f"{riot_id}#{riot_tag}" if riot_id and riot_tag else summoner_name
            
            if team_id == 100:  # Blue side
                blue_team.append(champion)
                blue_players.append(player_name)
            elif team_id == 200:  # Red side
                red_team.append(champion)
                red_players.append(player_name)
        
        # Sort by position (top=1, jungle=2, mid=3, adc=4, support=5)
        # Riot API provides teamPosition field
        def get_position_key(participant):
            position = participant.get('teamPosition', '')
            position_map = {'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'UTILITY': 4}
            return position_map.get(position, 99)
        
        # Reorder by position
        blue_participants = [p for p in participants if p.get('teamId') == 100]
        red_participants = [p for p in participants if p.get('teamId') == 200]
        
        blue_participants.sort(key=get_position_key)
        red_participants.sort(key=get_position_key)
        
        blue_team = [p.get('championName', '') for p in blue_participants]
        red_team = [p.get('championName', '') for p in red_participants]
        
        blue_players = []
        red_players = []
        for p in blue_participants:
            riot_id = p.get('riotIdGameName', '')
            riot_tag = p.get('riotIdTagline', '')
            player_name = f"{riot_id}#{riot_tag}" if riot_id and riot_tag else p.get('summonerName', '')
            blue_players.append(player_name)
        
        for p in red_participants:
            riot_id = p.get('riotIdGameName', '')
            riot_tag = p.get('riotIdTagline', '')
            player_name = f"{riot_id}#{riot_tag}" if riot_id and riot_tag else p.get('summonerName', '')
            red_players.append(player_name)
        
        return {
            'blue_team': blue_team,
            'red_team': red_team,
            'blue_players': blue_players,
            'red_players': red_players,
            'match_id': data.get('metadata', {}).get('matchId', ''),
            'region': info.get('platformId', ''),
            'url': None
        }
    
    def fetch_match_by_id(self, match_id: str, region: str = 'na1', use_riot_api: bool = False) -> Dict:
        """
        Fetch complete match data by match ID.
        
        Args:
            match_id: Match ID (Riot match ID if use_riot_api=True, otherwise u.gg identifier)
            region: Region code (default 'na1' for u.gg, 'americas' for Riot API)
            use_riot_api: If True, use Riot Games API; if False, try to fetch from u.gg
            
        Returns:
            Dictionary with blue_team, red_team, blue_players, red_players, and other match info
        """
        if use_riot_api:
            # Map common region codes to Riot API regions
            region_map = {
                'na1': 'americas', 'na': 'americas',
                'euw1': 'europe', 'euw': 'europe',
                'eun1': 'europe', 'eun': 'europe',
                'kr': 'asia', 'jp1': 'asia', 'jp': 'asia'
            }
            api_region = region_map.get(region.lower(), 'americas')
            return self.fetch_match_by_riot_id(match_id, api_region)
        else:
            # Try u.gg (may not work if they don't use match IDs)
            match_data = self._fetch_ugg_match_data(region, match_id)
            return match_data
    
    def calculate_complete_match_score(self, match_id: str = None, region: str = 'na1',
                                      w1: float = 0.35, w2: float = 0.35, w3: float = 0.3,
                                      use_riot_api: bool = False,
                                      blue_team: List[str] = None, red_team: List[str] = None,
                                      blue_players: List[str] = None, red_players: List[str] = None) -> Dict:
        """
        Calculate all three scores (lane matchups, player comfort, composition) for a match.
        
        Can use either:
        1. Riot Games API match ID (requires API key)
        2. Manual input of teams and players
        
        Args:
            match_id: Riot Games match ID (e.g., 'NA1_1234567890') or None for manual input
            region: Region code (default 'na1' for u.gg, 'americas' for Riot API)
            w1: Weight for lane matchups (default 0.35)
            w2: Weight for player comfort (default 0.35)
            w3: Weight for composition score (default 0.3)
            use_riot_api: If True, use Riot Games API; if False, try u.gg
            blue_team: Manual input - List of 5 blue side champions [top, jungle, mid, adc, support]
            red_team: Manual input - List of 5 red side champions [top, jungle, mid, adc, support]
            blue_players: Manual input - List of 5 blue side player names
            red_players: Manual input - List of 5 red side player names
            
        Returns:
            Dictionary with all scores and details
        """
        # Fetch match data or use manual input
        if blue_team and red_team:
            # Manual input
            match_data = {
                'blue_team': blue_team,
                'red_team': red_team,
                'blue_players': blue_players or [],
                'red_players': red_players or [],
                'match_id': 'manual',
                'region': region
            }
        elif match_id:
            # Fetch from API or u.gg
            match_data = self.fetch_match_by_id(match_id, region, use_riot_api=use_riot_api)
        else:
            raise ValueError("Either provide match_id or manually provide blue_team and red_team")
        
        blue_team = match_data['blue_team']
        red_team = match_data['red_team']
        blue_players = match_data.get('blue_players', [])
        red_players = match_data.get('red_players', [])
        
        if not blue_team or not red_team:
            raise ValueError("Could not extract team compositions from match data")
        
        results = {
            'match_id': match_id,
            'region': region,
            'blue_team': blue_team,
            'red_team': red_team,
            'blue_players': blue_players,
            'red_players': red_players
        }
        
        # Calculate lane matchup score
        matchup_details = self.calculate_lane_matchup_score(
            blue_team, red_team, None, w1, return_details=True
        )
        results['lane_matchup'] = matchup_details
        
        # Calculate player comfort score (if we have player names)
        if blue_players and red_players and len(blue_players) == 5 and len(red_players) == 5:
            comfort_details = self.calculate_player_comfort_score(
                blue_players, red_players,
                blue_team, red_team,
                region, w2, return_details=True
            )
            results['player_comfort'] = comfort_details
        else:
            results['player_comfort'] = {
                'score': 0.0,
                'note': 'Player names not available in match data'
            }
        
        # Calculate composition scores for both teams
        try:
            blue_comp = self.calculate_composition_score(blue_team, w3)
            red_comp = self.calculate_composition_score(red_team, w3)
            # Composition advantage = Blue comp - Red comp
            comp_advantage = blue_comp['total_score'] - red_comp['total_score']
            results['composition'] = {
                'blue_score': blue_comp['total_score'],
                'red_score': red_comp['total_score'],
                'advantage': comp_advantage,
                'w3': w3
            }
        except Exception as e:
            results['composition'] = {
                'error': str(e)
            }
        
        # Calculate S score: S = w1(LaneMatchups) + w2(Comfort) + w3(TeamComp)
        # Note: Scores are already weighted, so S is just the sum
        lane_matchup_score = matchup_details['score']
        player_comfort_score = results['player_comfort'].get('score', 0.0)
        composition_advantage = results['composition'].get('advantage', 0.0)
        
        S = lane_matchup_score + player_comfort_score + composition_advantage
        
        # Calculate U(Blue) = 1 / (1 + e^-(S + alpha)) using sigmoid function
        # This converts the score to win probability
        import math
        U_blue = 1 / (1 + math.exp(-(S + self.ALPHA)))
        
        results['S_score'] = S
        results['alpha'] = self.ALPHA
        results['U_blue'] = U_blue
        results['blue_win_probability'] = U_blue
        results['red_win_probability'] = 1 - U_blue
        
        # Keep total_payoff for backwards compatibility (clamped version)
        results['total_payoff'] = max(-0.15, min(0.15, S))
        
        return results
    
    def close(self):
        """Close database connection."""
        self.conn.close()


