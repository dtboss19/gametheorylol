import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import sqlite3
import time

BASE = "https://lol.fandom.com"
QUERY_URL = BASE + "/wiki/Special:RunQuery/TournamentStatistics"

# ============================================================================
# CHAMPION POOL - 5 most played per role + 1 extra each (30 total)
# Names must match lolchampiontags.db format (no spaces/apostrophes)
# NOTE: K'Sante not in DB, using Gnar instead
# ============================================================================
CHAMPION_POOL = {
    "TOP": ["Ksante", "Sion", "Rumble", "Ambessa", "Renekton", "RekSai"],
    "JUNGLE": ["XinZhao", "JarvanIV", "Wukong", "Trundle", "Vi", "Qiyana"],
    "MID": ["Ryze", "Taliyah", "Orianna", "Aurora", "Azir", "Galio"],
    "BOT": ["Corki", "KaiSa", "Smolder", "Caitlyn", "Ezreal", "Sivir"],
    "SUPPORT": ["Alistar", "Rakan", "Neeko", "Braum", "Leona", "Bard"],
}

# Extended champion pool (all champions from Worlds 2025)
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

# Flag to use extended pool (set to True to use extended, False for original)
USE_EXTENDED_POOL = False

# Select which pool to use based on flag
if USE_EXTENDED_POOL:
    CHAMPION_POOL = EXTENDED_CHAMPION_POOL

ALL_CHAMPIONS = [champ for role_champs in CHAMPION_POOL.values() for champ in role_champs]

# ============================================================================
# DEFAULT TEAM ROSTERS
# ============================================================================
TEAM_1 = {
    "name": "KT Rolster",
    "players": {
        "TOP": "PerfecT (Lee Seung-min)",      # KT top laner
        "JUNGLE": "Cuzz",
        "MID": "Bdd",
        "BOT": "deokdam",
        "SUPPORT": "Peter (Jeong Yoon-su)",    # KT support
    }
}

TEAM_2 = {
    "name": "T1",
    "players": {
        "TOP": "Doran (Choi Hyeon-joon)",         # T1 top laner
        "JUNGLE": "Oner",
        "MID": "Faker",
        "BOT": "Gumayusi",
        "SUPPORT": "Keria",
    }
}

# ============================================================================
# CHAMPION NAME NORMALIZATION
# Maps fandom.com names to lolchampiontags.db format
# ============================================================================
CHAMPION_NAME_MAP = {}  # Populated by load_champion_name_map()

def load_champion_name_map(db_path="lolchampiontags.db"):
    """
    Load champion names from lolchampiontags.db and create a lookup map.
    This allows us to match fandom names to database names.
    """
    global CHAMPION_NAME_MAP
    CHAMPION_NAME_MAP = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM champions")
        
        for (name,) in cursor.fetchall():
            # Store the exact name from DB
            CHAMPION_NAME_MAP[name.lower()] = name
            
            # Also store normalized versions for matching
            # Remove spaces and apostrophes for matching
            normalized = name.lower().replace(" ", "").replace("'", "")
            CHAMPION_NAME_MAP[normalized] = name
        
        conn.close()
        print(f"Loaded {len(CHAMPION_NAME_MAP)//2} champion names from {db_path}")
        return True
    except Exception as e:
        print(f"⚠️ Could not load champion names: {e}")
        return False


def normalize_champion_name(fandom_name: str) -> str:
    """
    Convert a champion name from fandom format to lolchampiontags.db format.
    
    Examples:
        "Jarvan IV" -> "JarvanIV"
        "Kai'Sa" -> "KaiSa"
        "Xin Zhao" -> "XinZhao"
        "K'Sante" -> "Ksante"
        "Rek'Sai" -> "RekSai"
        "Miss Fortune" -> "MissFortune"
        "Dr. Mundo" -> "DrMundo"
    """
    if not fandom_name or not isinstance(fandom_name, str):
        return fandom_name
    
    original = fandom_name.strip()
    
    # Skip summary/non-champion rows
    if original.lower() in ['overall:', 'total:', 'overall', 'total', '']:
        return None  # Will be filtered out
    
    # Load map if not already loaded
    if not CHAMPION_NAME_MAP:
        load_champion_name_map()
    
    # Special case mappings for problematic names
    # (Applied BEFORE standard normalization)
    special_mappings = {
        # K'Sante variations
        "k'sante": "Ksante",
        "ksante": "Ksante",
        # Dr. Mundo variations
        "dr. mundo": "DrMundo",
        "dr mundo": "DrMundo",
        "drmundo": "DrMundo",
        # Nunu variations
        "nunu & willump": "Nunu",
        "nunu": "Nunu",
        # Wukong variations
        "wukong": "Wukong",
        "monkeyking": "Wukong",
        # Renata variations
        "renata glasc": "Renata",
        "renataglasc": "Renata",
        "renata": "Renata",
    }
    
    # Check special mappings first (case-insensitive)
    original_lower = original.lower()
    if original_lower in special_mappings:
        return special_mappings[original_lower]
    
    # Try exact match (case-insensitive)
    if original_lower in CHAMPION_NAME_MAP:
        return CHAMPION_NAME_MAP[original_lower]
    
    # Normalize: remove spaces, apostrophes, and periods
    normalized = original.replace(" ", "").replace("'", "").replace(".", "").lower()
    
    # Check special mappings again with normalized form
    if normalized in special_mappings:
        return special_mappings[normalized]
    
    if normalized in CHAMPION_NAME_MAP:
        return CHAMPION_NAME_MAP[normalized]
    
    # If no match found, return with basic normalization
    result = original.replace("'", "").replace(" ", "").replace(".", "")
    print(f"   ⚠️ Unknown champion: '{original}' -> '{result}' (not in DB)")
    return result


def normalize_dataframe_champions(df: pd.DataFrame, champion_column: str = "Champion") -> pd.DataFrame:
    """
    Normalize all champion names in a DataFrame to match lolchampiontags.db format.
    Also removes summary rows (Overall, Total, etc.)
    """
    if champion_column not in df.columns:
        return df
    
    # Apply normalization to the champion column
    df[champion_column] = df[champion_column].apply(normalize_champion_name)
    
    # Remove rows where champion is None (summary rows)
    df = df[df[champion_column].notna()]
    df = df[df[champion_column] != '']
    
    return df


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================
def run_query(player_name, year="2025", tournament=""):
    """Fetch raw HTML from lol.fandom.com TournamentStatistics query."""
    params = {
        "TS[preload]": "PlayerByChampion",
        "TS[tournament]": tournament,
        "TS[link]": player_name,
        "TS[champion]": "",
        "TS[role]": "",
        "TS[team]": "",
        "TS[patch]": "",
        "TS[year]": year,
        "TS[region]": "",
        "TS[tournamentlevel]": "",
        "TS[where]": "",
        "TS[includelink][is_checkbox]": "true",
        "TS[shownet][is_checkbox]": "true",
        "_run": "",
        "pfRunQueryFormName": "TournamentStatistics",
        "wpRunQuery": "",
        "pf_free_text": ""
    }
    resp = requests.get(QUERY_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_query_html(html):
    """Parse the wikitable from HTML and return a flat DataFrame with stats."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "wikitable"})
    if table is None:
        return None

    # Read HTML table
    df = pd.read_html(StringIO(str(table)))[0]
    
    # Flatten MultiIndex columns - extract just the LAST level (actual column names)
    if isinstance(df.columns, pd.MultiIndex):
        # The last level has the actual column names: 'Champion', 'G', 'W', 'L', 'WR', etc.
        df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns.values]
    
    # Map short column names to full names
    column_mapping = {
        'Champion': 'Champion',
        'G': 'Games',
        'W': 'Wins',
        'L': 'Losses',
        'WR': 'WinRate',
        'K': 'Kills',
        'D': 'Deaths',
        'A': 'Assists',
        'KDA': 'KDA',
        'CS': 'CS',
        'CS/M': 'CS_per_min',
        'G.1': 'Gold',  # G.1 is Gold (second 'G' column)
        'G/M': 'Gold_per_min',
        'DMG': 'Damage',
        'DMG/M': 'Damage_per_min',
        'KPAR': 'KillParticipation',
        'KS': 'KillShare',
        'GS': 'GoldShare',
        'VS': 'VisionScore',
        'VS/M': 'VisionScore_per_min',
    }
    
    # Apply column mapping
    new_columns = []
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in column_mapping:
            new_columns.append(column_mapping[col_str])
        else:
            new_columns.append(col_str)
    
    # Deduplicate column names by keeping only the first occurrence
    # (The first Champion column has the actual data, others are empty)
    seen_cols = {}
    final_columns = []
    cols_to_drop = []
    
    for i, col in enumerate(new_columns):
        if col in seen_cols:
            # Mark duplicate column for removal
            cols_to_drop.append(i)
            final_columns.append(f"{col}_{seen_cols[col]}")
            seen_cols[col] += 1
        else:
            seen_cols[col] = 1
            final_columns.append(col)
    
    df.columns = final_columns
    
    # Drop duplicate columns (keep first occurrence)
    if cols_to_drop:
        df = df.drop(df.columns[cols_to_drop], axis=1)
    
    return df


def get_player_champion_stats(player_name, year="2025"):
    """Fetch and parse champion stats for a single player."""
    print(f"  Fetching {player_name}...")
    html = run_query(player_name, year=year)
    df = parse_query_html(html)
    
    if df is None or df.empty:
        print(f"  ⚠ No data for {player_name}")
        return None

    df["Player"] = player_name
    
    # Try to extract win rate percentage
    wr_cols = [c for c in df.columns if 'WR' in c.upper() or 'WIN' in c.upper()]
    for col in wr_cols:
        if df[col].dtype == object:
            try:
                df["WR_pct"] = df[col].str.rstrip("%").astype(float)
                break
            except (ValueError, AttributeError):
                continue
    
    return df


def scrape_team(team_dict, year="2025"):
    """Scrape champion stats for all players on a team."""
    print(f"\nScraping {team_dict['name']}...")
    results = []
    
    for role, player in team_dict["players"].items():
        df = get_player_champion_stats(player, year=year)
        if df is not None:
            df["Role"] = role
            df["Team"] = team_dict["name"]
            results.append(df)
        time.sleep(1)  # Rate limiting
    
    if not results:
        print(f"  [ERROR] No data found for {team_dict['name']}")
        return None
    
    # Concatenate with sort=False to preserve order
    combined = pd.concat(results, ignore_index=True, sort=False)
    
    # Normalize champion names to match lolchampiontags.db format
    print(f"  Normalizing champion names...")
    combined = normalize_dataframe_champions(combined, "Champion")
    
    print(f"  Got {len(combined)} champion records for {team_dict['name']}")
    return combined


def save_to_sqlite(df, db_name="worlds2025.db", table="champion_stats"):
    """Save DataFrame to SQLite database."""
    conn = sqlite3.connect(db_name)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"\n💾 Saved to {db_name} (table: {table})")


def save_to_csv(df, filename="worlds2025_stats.csv"):
    """Save DataFrame to CSV."""
    df.to_csv(filename, index=False)
    print(f"📄 Saved to {filename}")


# ============================================================================
# PLAYER COMFORT DATA - Stores player winrates on champions
# ============================================================================
PLAYER_COMFORT_DATA = {}  # Populated by load_player_comfort_data()

def load_player_comfort_data(db_name="worlds2025.db"):
    """
    Load player champion stats from the database into PLAYER_COMFORT_DATA.
    Structure: {player_name: {champion_name: {'games': N, 'wins': N, 'winrate': 0.XX}}}
    """
    global PLAYER_COMFORT_DATA
    PLAYER_COMFORT_DATA = {}
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Query all player-champion stats
        cursor.execute("""
            SELECT Player, Champion, Games, Wins, WinRate 
            FROM champion_stats 
            WHERE Champion IS NOT NULL
        """)
        
        for row in cursor.fetchall():
            player, champion, games, wins, winrate = row
            if player not in PLAYER_COMFORT_DATA:
                PLAYER_COMFORT_DATA[player] = {}
            
            # Parse winrate if it's a string percentage
            if isinstance(winrate, str) and '%' in winrate:
                winrate = float(winrate.rstrip('%')) / 100.0
            elif winrate is None:
                winrate = 0.5
            else:
                try:
                    winrate = float(winrate)
                    if winrate > 1:  # If it's a percentage like 55.5
                        winrate = winrate / 100.0
                except:
                    winrate = 0.5
            
            PLAYER_COMFORT_DATA[player][champion] = {
                'games': int(games) if games else 0,
                'wins': int(wins) if wins else 0,
                'winrate': winrate
            }
        
        conn.close()
        print(f"Loaded comfort data for {len(PLAYER_COMFORT_DATA)} players")
        return True
    except Exception as e:
        print(f"⚠️ Could not load player comfort data: {e}")
        return False


def get_player_comfort(player_name, champion_name, min_games=5):
    """
    Get a player's winrate on a specific champion.
    Returns 0.5 (neutral) if not enough games or data not found.
    """
    if player_name not in PLAYER_COMFORT_DATA:
        return {'winrate': 0.5, 'games': 0, 'found': False}
    
    player_data = PLAYER_COMFORT_DATA[player_name]
    
    # Try exact match first
    if champion_name in player_data:
        data = player_data[champion_name]
        if data['games'] >= min_games:
            return {'winrate': data['winrate'], 'games': data['games'], 'found': True}
        else:
            return {'winrate': 0.5, 'games': data['games'], 'found': True}
    
    # Try case-insensitive match
    for champ, data in player_data.items():
        if champ.lower() == champion_name.lower():
            if data['games'] >= min_games:
                return {'winrate': data['winrate'], 'games': data['games'], 'found': True}
            else:
                return {'winrate': 0.5, 'games': data['games'], 'found': True}
    
    return {'winrate': 0.5, 'games': 0, 'found': False}


# ============================================================================
# DRAFT ANALYSIS STRUCTURE - Role-based picks
# ============================================================================
ROLE_ORDER = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]

class DraftState:
    """
    Represents the current state of a draft with ROLE-BASED tracking.
    Each pick is assigned to a specific role slot.
    
    Supports sequential draft where:
    - Blue (Team 1) has 2 open roles (picks both before Red's final pick)
    - Red (Team 2) has 1 open role (picks last after seeing Blue's picks)
    """
    
    def __init__(self):
        self.bans = []  # 10 banned champions
        # Role -> Champion mapping (None means role not picked yet)
        self.team1_picks = {role: None for role in ROLE_ORDER}
        self.team2_picks = {role: None for role in ROLE_ORDER}
        self.team1_last_role = None  # Single open role (legacy)
        self.team1_open_roles = []   # Multiple open roles for Blue
        self.team2_last_role = None  # Single open role for Red
        
    def get_available_champions(self, role=None):
        """Get champions not banned or picked."""
        # Get all picked champions from both teams
        team1_champs = [c for c in self.team1_picks.values() if c is not None]
        team2_champs = [c for c in self.team2_picks.values() if c is not None]
        taken = set(self.bans + team1_champs + team2_champs)
        
        if role:
            return [c for c in CHAMPION_POOL.get(role, []) if c not in taken]
        return [c for c in ALL_CHAMPIONS if c not in taken]
    
    def set_bans(self, bans):
        """Set the 10 banned champions."""
        self.bans = bans
        
    def set_team1_picks(self, role_picks: dict, last_pick_role: str):
        """
        Set team 1's picks by role.
        Args:
            role_picks: Dict mapping role -> champion (e.g., {"TOP": "Sion", "MID": "Ryze"})
            last_pick_role: The role that still needs to be picked
        """
        for role, champ in role_picks.items():
            if role in self.team1_picks:
                self.team1_picks[role] = champ
        self.team1_last_role = last_pick_role
        
    def set_team2_picks(self, role_picks: dict, last_pick_role: str):
        """
        Set team 2's picks by role.
        Args:
            role_picks: Dict mapping role -> champion (e.g., {"TOP": "Rumble", "JUNGLE": "Wukong"})
            last_pick_role: The role that still needs to be picked
        """
        for role, champ in role_picks.items():
            if role in self.team2_picks:
                self.team2_picks[role] = champ
        self.team2_last_role = last_pick_role
    
    def get_team1_as_list(self):
        """Get team 1 picks as ordered list [TOP, JG, MID, BOT, SUP]."""
        return [self.team1_picks.get(role) for role in ROLE_ORDER]
    
    def get_team2_as_list(self):
        """Get team 2 picks as ordered list [TOP, JG, MID, BOT, SUP]."""
        return [self.team2_picks.get(role) for role in ROLE_ORDER]
    
    def get_complete_team1(self, last_pick_champ):
        """Get complete team 1 with the last pick filled in (single role)."""
        picks = self.team1_picks.copy()
        picks[self.team1_last_role] = last_pick_champ
        return [picks.get(role) for role in ROLE_ORDER]
    
    def get_complete_team1_multi(self, role_picks: dict):
        """
        Get complete team 1 with multiple open roles filled in.
        Args:
            role_picks: Dict mapping open roles to champions (e.g., {"MID": "Azir", "SUPPORT": "Alistar"})
        """
        picks = self.team1_picks.copy()
        for role, champ in role_picks.items():
            picks[role] = champ
        return [picks.get(role) for role in ROLE_ORDER]
    
    def get_complete_team2(self, last_pick_champ):
        """Get complete team 2 with the last pick filled in."""
        picks = self.team2_picks.copy()
        picks[self.team2_last_role] = last_pick_champ
        return [picks.get(role) for role in ROLE_ORDER]
        
    def __str__(self):
        team1_list = self.get_team1_as_list()
        team2_list = self.get_team2_as_list()
        
        return f"""
Draft State:
  Bans: {self.bans}
  
  Team 1 Picks (by role):
    TOP:     {self.team1_picks['TOP']}
    JUNGLE:  {self.team1_picks['JUNGLE']}
    MID:     {self.team1_picks['MID']}
    BOT:     {self.team1_picks['BOT']}
    SUPPORT: {self.team1_picks['SUPPORT']}
  Last pick role: {self.team1_last_role}
  
  Team 2 Picks (by role):
    TOP:     {self.team2_picks['TOP']}
    JUNGLE:  {self.team2_picks['JUNGLE']}
    MID:     {self.team2_picks['MID']}
    BOT:     {self.team2_picks['BOT']}
    SUPPORT: {self.team2_picks['SUPPORT']}
  Last pick role: {self.team2_last_role}
  
  Available for Team 1 ({self.team1_last_role}): {self.get_available_champions(self.team1_last_role)}
  Available for Team 2 ({self.team2_last_role}): {self.get_available_champions(self.team2_last_role)}
"""


def create_example_draft():
    """Create an example draft scenario with ROLE-BASED picks."""
    draft = DraftState()
    
    # Example: Ban 10 champions (2 from each role)
    draft.set_bans([
        "Gnar", "Ambessa",     # TOP bans
        "XinZhao", "Vi",       # JG bans
        "Aurora", "Azir",      # MID bans
        "Corki", "KaiSa",      # BOT bans
        "Alistar", "Rakan"     # SUP bans
    ])
    
    # Team 1 (KT) locks 4 picks BY ROLE, last pick is SUPPORT
    draft.set_team1_picks({
        "TOP": "Sion",
        "JUNGLE": "JarvanIV",
        "MID": "Ryze",
        "BOT": "Caitlyn",
        # SUPPORT is the last pick role
    }, last_pick_role="SUPPORT")
    
    # Team 2 (T1) locks 4 picks BY ROLE, last pick is MID
    draft.set_team2_picks({
        "TOP": "Rumble",
        "JUNGLE": "Wukong",
        # MID is the last pick role
        "BOT": "Ezreal",
        "SUPPORT": "Leona",
    }, last_pick_role="MID")
    
    return draft


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("WORLDS 2025 DRAFT ANALYSIS DATA FETCHER")
    print("=" * 60)
    
    print("\nChampion Pool:")
    for role, champs in CHAMPION_POOL.items():
        print(f"  {role}: {', '.join(champs)}")
    
    print(f"\n  Total: {len(ALL_CHAMPIONS)} champions")
    
    # Load champion name mapping for normalization
    print("\n" + "=" * 60)
    print("LOADING CHAMPION NAME MAPPING")
    print("=" * 60)
    load_champion_name_map()
    
    # Fetch player data
    print("\n" + "=" * 60)
    print("FETCHING PLAYER DATA")
    print("=" * 60)
    
    kt_df = scrape_team(TEAM_1)
    t1_df = scrape_team(TEAM_2)
    
    # Combine and save
    dfs_to_concat = [d for d in [kt_df, t1_df] if d is not None]
    
    if dfs_to_concat:
        all_df = pd.concat(dfs_to_concat, ignore_index=True, sort=False)
        print(f"\nTotal records: {len(all_df)}")
        
        # Print column names for debugging
        print(f"\nColumns: {list(all_df.columns)}")
        
        # Show sample of normalized champion names
        if 'Champion' in all_df.columns:
            unique_champs = all_df['Champion'].unique()
            print(f"\n🎮 Unique Champions ({len(unique_champs)}): {list(unique_champs)[:15]}...")
        
        save_to_sqlite(all_df)
        save_to_csv(all_df)
    else:
        print("\n[ERROR] No data fetched!")
    
    # Show example draft
    print("\n" + "=" * 60)
    print("EXAMPLE DRAFT SCENARIO")
    print("=" * 60)
    
    draft = create_example_draft()
    print(draft)
