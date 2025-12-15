"""
Test script for Worlds 2025 draft scenarios with extended champion pool.
Supports 5 predefined game scenarios with specific picks and bans.
"""

from get_worlds_data import DraftState, ROLE_ORDER, load_champion_name_map, normalize_champion_name
import sqlite3

# ============================================================================
# EXTENDED CHAMPION POOL - All champions from Worlds 2025
# Names must match lolchampiontags.db format (no spaces/apostrophes)
# ============================================================================
EXTENDED_CHAMPION_POOL = {
    "TOP": [
        "Ksante", "Sion", "Rumble", "Ambessa", "RekSai", "Renekton", 
        "Ornn", "Mordekaiser", "Aatrox", "Jax", "Poppy", "Camille", 
        "Galio", "Yorick", "Kled", "Jayce", "Gwen", "Gragas", "Aurora", "Gnar"
    ],
    "JUNGLE": [
        "XinZhao", "JarvanIV", "Wukong", "Trundle", "Vi", "Qiyana", 
        "Pantheon", "Naafiri", "DrMundo", "Nocturne", "Sejuani", "Skarner", 
        "Poppy", "Viego", "Maokai", "Nidalee"
    ],
    "MID": [
        "Ryze", "Orianna", "Taliyah", "Aurora", "Azir", "Galio", 
        "Cassiopeia", "Viktor", "Yone", "Akali", "Annie", "Sylas", 
        "Anivia", "LeBlanc", "Hwei", "Syndra", "Mel", "Ziggs", "Smolder", 
        "Ahri", "Swain", "Zoe"
    ],
    "BOT": [
        "Corki", "KaiSa", "Yunara", "Caitlyn", "Ezreal", "Sivir", 
        "Varus", "Xayah", "Lucian", "Ziggs", "Draven", "Jhin", "Ashe", 
        "Kalista", "Vayne", "MissFortune", "Smolder", "Jinx"
    ],
    "SUPPORT": [
        "Alistar", "Rakan", "Neeko", "Braum", "Leona", "Bard", 
        "Nautilus", "Rell", "Poppy", "Karma", "Nami", "Lulu", 
        "Renata", "TahmKench"
    ],
}

ALL_EXTENDED_CHAMPIONS = [champ for role_champs in EXTENDED_CHAMPION_POOL.values() for champ in role_champs]


class ExtendedDraftState(DraftState):
    """
    Extended DraftState that uses the extended champion pool.
    """
    
    def get_available_champions(self, role=None):
        """Get champions not banned or picked from extended pool."""
        # Get all picked champions from both teams
        team1_champs = [c for c in self.team1_picks.values() if c is not None]
        team2_champs = [c for c in self.team2_picks.values() if c is not None]
        taken = set(self.bans + team1_champs + team2_champs)
        
        if role:
            return [c for c in EXTENDED_CHAMPION_POOL.get(role, []) if c not in taken]
        return [c for c in ALL_EXTENDED_CHAMPIONS if c not in taken]


def create_draft_from_scenario(bans, team1_picks, team1_last_role, team2_picks, team2_last_role):
    """
    Create a DraftState from a specific scenario.
    
    Args:
        bans: List of 10 banned champions
        team1_picks: Dict mapping role -> champion for team 1 (e.g., {"TOP": "Sion", "JUNGLE": "JarvanIV"})
        team1_last_role: The role that team 1 still needs to pick
        team2_picks: Dict mapping role -> champion for team 2
        team2_last_role: The role that team 2 still needs to pick
    
    Returns:
        ExtendedDraftState object
    """
    draft = ExtendedDraftState()
    
    # Normalize champion names
    load_champion_name_map()
    normalized_bans = [normalize_champion_name(ban) for ban in bans if normalize_champion_name(ban)]
    
    normalized_team1_picks = {}
    for role, champ in team1_picks.items():
        normalized = normalize_champion_name(champ)
        if normalized:
            normalized_team1_picks[role] = normalized
    
    normalized_team2_picks = {}
    for role, champ in team2_picks.items():
        normalized = normalize_champion_name(champ)
        if normalized:
            normalized_team2_picks[role] = normalized
    
    draft.set_bans(normalized_bans)
    draft.set_team1_picks(normalized_team1_picks, team1_last_role)
    draft.set_team2_picks(normalized_team2_picks, team2_last_role)
    
    return draft


# ============================================================================
# GAME SCENARIOS - 5 predefined scenarios
# ============================================================================

GAME_1 = {
    "bans": [
        "Yunara", "Azir",      # High priority bans
        "Poppy", "Neeko",       # Support/Jungle bans
        "Orianna", "Ryze",      # Mid bans
        "XinZhao", "JarvanIV",  # Jungle bans
        "Corki", "KaiSa"       # Bot bans
    ],
    "team1_picks": {
        "TOP": "Sion",
        "JUNGLE": "Wukong",
        "MID": "Taliyah",
        "BOT": "Caitlyn",
        # SUPPORT is last pick
    },
    "team1_last_role": "SUPPORT",
    "team2_picks": {
        "TOP": "Rumble",
        "JUNGLE": "Trundle",
        "MID": "Aurora",
        "SUPPORT": "Alistar",
        # BOT is last pick
    },
    "team2_last_role": "BOT"
}

GAME_2 = {
    "bans": [
        "Aurora", "Galio",      # Mid/Top bans
        "RekSai", "Renekton",   # Top bans
        "Vi", "Qiyana",         # Jungle bans
        "Smolder", "Varus",     # Bot bans
        "Rakan", "Bard"         # Support bans
    ],
    "team1_picks": {
        "TOP": "Ksante",
        "JUNGLE": "JarvanIV",
        "BOT": "Ezreal",
        "SUPPORT": "Leona",
        # MID is last pick
    },
    "team1_last_role": "MID",
    "team2_picks": {
        "TOP": "Ambessa",
        "JUNGLE": "XinZhao",
        "MID": "Ryze",
        "BOT": "Sivir",
        # SUPPORT is last pick
    },
    "team2_last_role": "SUPPORT"
}

GAME_3 = {
    "bans": [
        "Gwen", "Jax",          # Top bans
        "Pantheon", "Nocturne", # Jungle bans
        "Azir", "Orianna",      # Mid bans
        "Xayah", "Lucian",      # Bot bans
        "Neeko", "Braum"        # Support bans
    ],
    "team1_picks": {
        "TOP": "Rumble",
        "JUNGLE": "Wukong",
        "MID": "Taliyah",
        "SUPPORT": "Alistar",
        # BOT is last pick
    },
    "team1_last_role": "BOT",
    "team2_picks": {
        "TOP": "Sion",
        "JUNGLE": "Trundle",
        "MID": "Aurora",
        "BOT": "Corki",
        # SUPPORT is last pick
    },
    "team2_last_role": "SUPPORT"
}

GAME_4 = {
    "bans": [
        "Yorick", "Ornn",       # Top bans
        "Skarner", "Sejuani",   # Jungle bans
        "Viktor", "Syndra",     # Mid bans
        "Jhin", "Ashe",         # Bot bans
        "Nautilus", "Rell"      # Support bans
    ],
    "team1_picks": {
        "TOP": "Renekton",
        "JUNGLE": "JarvanIV",
        "MID": "Ryze",
        "BOT": "KaiSa",
        # SUPPORT is last pick
    },
    "team1_last_role": "SUPPORT",
    "team2_picks": {
        "TOP": "Gwen",
        "JUNGLE": "XinZhao",
        "MID": "Azir",
        "SUPPORT": "Rakan",
        # BOT is last pick
    },
    "team2_last_role": "BOT"
}

GAME_5 = {
    "bans": [
        "Camille", "Aatrox",    # Top bans
        "Viego", "Maokai",      # Jungle bans
        "Yone", "Akali",        # Mid bans
        "Kalista", "MissFortune", # Bot bans
        "Karma", "Lulu"         # Support bans
    ],
    "team1_picks": {
        "TOP": "Ambessa",
        "JUNGLE": "Wukong",
        "MID": "Orianna",
        "BOT": "Varus",
        # SUPPORT is last pick
    },
    "team1_last_role": "SUPPORT",
    "team2_picks": {
        "TOP": "Rumble",
        "JUNGLE": "Trundle",
        "MID": "Taliyah",
        "BOT": "Ezreal",
        # SUPPORT is last pick
    },
    "team2_last_role": "SUPPORT"
}

# All game scenarios
GAME_SCENARIOS = {
    "game_1": GAME_1,
    "game_2": GAME_2,
    "game_3": GAME_3,
    "game_4": GAME_4,
    "game_5": GAME_5,
}


def create_draft_for_game(game_number: int):
    """
    Create a draft state for a specific game (1-5).
    
    Args:
        game_number: Game number (1-5)
    
    Returns:
        ExtendedDraftState object
    """
    if game_number < 1 or game_number > 5:
        raise ValueError("Game number must be between 1 and 5")
    
    game_key = f"game_{game_number}"
    scenario = GAME_SCENARIOS[game_key]
    
    return create_draft_from_scenario(
        bans=scenario["bans"],
        team1_picks=scenario["team1_picks"],
        team1_last_role=scenario["team1_last_role"],
        team2_picks=scenario["team2_picks"],
        team2_last_role=scenario["team2_last_role"]
    )


def print_game_scenario(game_number: int):
    """Print a formatted view of a game scenario."""
    draft = create_draft_for_game(game_number)
    print(f"\n{'='*70}")
    print(f"GAME {game_number} SCENARIO")
    print(f"{'='*70}")
    print(draft)


def print_all_scenarios():
    """Print all 5 game scenarios."""
    for i in range(1, 6):
        print_game_scenario(i)
        print()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("WORLDS 2025 EXTENDED CHAMPION POOL - GAME SCENARIOS")
    print("=" * 70)
    
    print("\nExtended Champion Pool:")
    for role, champs in EXTENDED_CHAMPION_POOL.items():
        print(f"  {role}: {len(champs)} champions")
        print(f"    {', '.join(champs[:10])}{'...' if len(champs) > 10 else ''}")
    
    print(f"\n  Total: {len(ALL_EXTENDED_CHAMPIONS)} champions")
    
    # Load champion name mapping
    print("\n" + "=" * 70)
    print("LOADING CHAMPION NAME MAPPING")
    print("=" * 70)
    load_champion_name_map()
    
    # Display all scenarios
    print("\n" + "=" * 70)
    print("ALL GAME SCENARIOS")
    print("=" * 70)
    print_all_scenarios()
    
    # Example: Get available champions for a specific scenario
    print("\n" + "=" * 70)
    print("EXAMPLE: Available Champions for Game 1")
    print("=" * 70)
    draft = create_draft_for_game(1)
    print(f"\nTeam 1 ({draft.team1_last_role}): {draft.get_available_champions(draft.team1_last_role)}")
    print(f"Team 2 ({draft.team2_last_role}): {draft.get_available_champions(draft.team2_last_role)}")

