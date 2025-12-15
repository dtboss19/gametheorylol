# League of Legends Game Theory Solver

A Python implementation of a Subgame Perfect Nash Equilibrium (SPNE) solver for League of Legends draft phase analysis. The solver uses backward induction on realistic Worlds-style drafts, combining lane matchups, player comfort, and team composition into a single payoff, then finding optimal picks and best responses.

## Features

- **Subgame Perfect Nash Equilibrium (SPNE) Calculation**: Uses backward induction to find optimal draft strategies for both teams.
- **Team Composition Scoring**: Evaluates team compositions based on synergies, role requirements, and strategic elements.
- **Lane Matchup Analysis**: Calculates win rates between champions in specific roles, with heavy caching of matchup queries.
- **Player Comfort Integration**: Incorporates player-specific champion performance data from a SQLite database (with OP.GG scraping as fallback).
- **Role-Based Champion Selection**: Ensures teams fulfill all required roles (TOP, JUNGLE, MID, BOT, SUPPORT), using curated extended role pools.
- **Optimized Performance**: Uses memoization, role-specific pre-calculations, and caching for comfort and matchup data to make exhaustive searches tractable.

## Worlds 2025 Case Study: 5 Tested Games

This repository includes five end-to-end case studies that replicate specific KT vs T1 draft states from a hypothetical Worlds 2025 series. Each script computes SPNE picks, best responses, and compares them to the actual stage drafts:

- **`test_game1_nash.py`**:  
  - KT as Blue, T1 as Red.  
  - Blue chooses BOT + SUPPORT; Red chooses TOP.  
  - Pre-calculates Rumble vs all TOPs, Varus vs all BOTs, Poppy vs all SUPPORTs, then explores all open combinations.

- **`test_game2_nash.py`**:  
  - KT as Blue, T1 as Red.  
  - Blue chooses MID + BOT; Red chooses MID.  
  - Pre-calculates Reksai vs all TOPs, Sivir vs all BOTs, Lulu vs all SUPPORTs, then explores all MID vs MID combinations.

- **`test_game3_nash.py`**:  
  - T1 as Blue, KT as Red.  
  - Blue chooses JUNGLE + MID; Red chooses MID.  
  - DrMundo is locked JUNGLE for KT; the model pre-calculates DrMundo vs all JUNGLE champions and then explores MID vs MID.

- **`test_game4_nash.py`**:  
  - T1 as Blue, KT as Red.  
  - Blue chooses TOP + MID; Red chooses TOP.  
  - Cassiopeia is locked MID for KT; the model pre-calculates Cassiopeia vs all mids and explores all TOP vs TOP matchups as the main variable.

- **`test_game5_nash.py`**:  
  - KT as Blue, T1 as Red.  
  - KT chooses JUNGLE + MID; T1 chooses SUPPORT.  
  - The model pre-calculates all KT JUNGLE vs Pantheon (locked JUNGLE) and all KT MID vs Galio (locked MID), then explores all SUPPORT responses for T1.

For each game:

- The script prints:
  - The current draft state (bans, locked picks, open roles).
  - A full table of all blue-side combinations in the open roles, with:
    - Payoff from blue’s perspective.
    - Implied win probabilities for Blue and Red.
    - The red-side best response champion for the remaining role.
    - SPNE combination marked explicitly.
  - Scenario comparisons: optimal vs optimal (SPNE), actual vs actual (what was played), and cross scenarios (optimal vs actual, actual vs optimal).
  - A “Key Findings” section explaining how far the actual draft deviates from SPNE.

These 5 scripts are a complete, reproducible case study of how the model behaves on realistic, constrained game states.

## How the Model Works (High Level)

- **Backward Induction / SPNE**:
  - The solver in `nash_equilibrium.py` represents the draft as a sequence of picks.
  - It explores all legal picks for the current team (respecting bans, previous picks, and role constraints), recurses, and backs up payoffs from terminal states.
  - For the last few picks in each game-specific script (`test_game1_nash.py`–`test_game5_nash.py`), we also run an explicit minimax over all remaining combinations to verify and interpret the SPNE.

- **Payoff Function**:
  - Implemented in `SubgamePerfectNashEquilibrium.calculate_payoff` using `CompositionScorer`.
  - Combines three components:

    ```
    S = w1 * LaneMatchups + w2 * Comfort + w3 * TeamComp
    ```

  - **LaneMatchups**: Per-lane win rate differences from u.gg (cached via `matchup_cache`), with role-specific pre-calcs (e.g., Rumble vs all TOPs, Cassiopeia vs all mids).
  - **Comfort**: Player–champion winrates from `worlds2025.db` with OP.GG scraping as fallback, cached in `player_comfort_cache`.
  - **TeamComp**: Composition attributes (engage, peel, damage profile, etc.) from champion tags and role data.
  - The scalar payoff is converted to a win probability using a logistic transform; positive payoff favors Blue, negative favors Red.

- **Caching and Performance**:
  - **Payoff caching**: Entire team comps are memoized so re-evaluating the same 5v5 is O(1).
  - **Matchup caching**: u.gg matchup results are cached once per pair (per role) and reused.
  - **Comfort caching**: Player–champion comfort values are cached, and database lookups are preferred over scraping.
  - The game-specific tests also run pre-calculation passes for locked matchups so that exhaustive searches over the remaining open roles are fast enough to run locally.

## Project Structure

```
gametheorylol/
├── nash_equilibrium.py         # SPNE solver implementation
├── composition_scorer.py       # Team composition and matchup scoring
├── get_worlds_data.py          # Worlds 2025 DB + comfort helpers
├── test_worlds2025_scenarios.py# Extended champion pools and scenarios
├── test_game1_nash.py          # Game 1 KT vs T1 analysis (BOT+SUP vs TOP)
├── test_game2_nash.py          # Game 2 KT vs T1 analysis (MID+BOT vs MID)
├── test_game3_nash.py          # Game 3 T1 vs KT analysis (JUNGLE+MID vs MID)
├── test_game4_nash.py          # Game 4 T1 vs KT analysis (TOP+MID vs TOP)
├── test_game5_nash.py          # Game 5 KT vs T1 analysis (JUNGLE+MID vs SUPPORT)
├── benchmark_spne.py           # Performance benchmarking tools
├── import_roles.py             # Database import utilities
├── test_nash_equilibrium.py    # Generic SPNE tests
├── test_complete_match.py      # Complete match analysis tests
├── test_composition.py         # Composition scoring tests
├── test_matchup.py             # Matchup analysis tests
├── test_player_comfort.py      # Player comfort tests
├── test_riot_api.py            # Riot API integration tests
├── test_opgg_scraping.py       # OP.GG scraping tests
├── test_winrate_extraction.py  # Winrate extraction tests
├── requirements.txt            # Python dependencies
├── lolchampiontags.db         # Main champion database
└── README.md                  
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/gametheorylol.git
cd gametheorylol
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up the databases:
   - Ensure `lolchampiontags.db` exists in the project root.
   - Ensure `worlds2025.db` (or equivalent) exists if you want player comfort data from Worlds.
   - Run `import_roles.py` if you need to import or rebuild role/tag data.

4. (Optional) Set up Riot API / scraping keys for live comfort data:
   - Create a `.env` file.
   - Add: `RIOT_API_KEY=your_api_key_here` (if applicable for your comfort data pipeline).

## Usage

### 1. Basic SPNE Calculation (Generic Draft)

```python
from nash_equilibrium import SubgamePerfectNashEquilibrium
from composition_scorer import CompositionScorer

# Initialize
scorer = CompositionScorer()
spne = SubgamePerfectNashEquilibrium(
    scorer,
    w1=0.15,  # Weight for lane matchups
    w2=0.25,  # Weight for player comfort
    w3=0.60,  # Weight for composition score
    skip_api_calls=True,  # Set to False for accurate matchups
    show_progress=True
)

# Get available champions
all_champions = spne.get_all_champions()

# Define draft order (standard League draft)
draft_order = ['blue', 'red', 'red', 'blue', 'blue',
               'red', 'red', 'blue', 'blue', 'red']

# Find SPNE
result = spne.find_spne_sequential_draft(
    available_champions=all_champions[:50],  # Limit for performance
    draft_order=draft_order,
    max_depth=5,  # Explore 5 picks via backward induction
    auto_ban_counters=True,  # Auto-ban counter champions
    bans_per_pick=1  # Number of bans per pick
)

print(f"Blue Team: {result['blue_team']}")
print(f"Red Team: {result['red_team']}")
print(f"Payoff (Blue advantage): {result['payoff']:.4f}")
```

### 2. Running the 5 Worlds 2025 Game Analyses

Each game script is executable on its own and prints a complete report to stdout:

```bash
# Game 1: KT (Blue) vs T1 (Red)
python test_game1_nash.py

# Game 2: KT (Blue) vs T1 (Red)
python test_game2_nash.py

# Game 3: T1 (Blue) vs KT (Red)
python test_game3_nash.py

# Game 4: T1 (Blue) vs KT (Red)
python test_game4_nash.py

# Game 5: KT (Blue) vs T1 (Red)
python test_game5_nash.py
```

Each script:

- Creates a draft state from bans and locked picks.
- Uses the SPNE solver and `CompositionScorer` with memoization enabled.
- Runs an explicit minimax over the final open roles to:
  - Enumerate all candidate blue-side combinations from the relevant extended pools.
  - Enumerate all candidate red-side best responses in the remaining role.
  - Identify and label the SPNE combination and its payoff.
  - Compare actual vs optimal choices and report win-rate differences.

### 3. Running Generic Tests and Benchmarks

```bash
# Run main SPNE test
python test_nash_equilibrium.py

# Run specific SPNE mode
python test_nash_equilibrium.py sequential

# Run composition tests
python test_composition.py

# Run complete match analysis
python test_complete_match.py

# Benchmark solver performance
python benchmark_spne.py
```

## Configuration Options

### SPNE Solver Parameters

- `w1` (default: 0.15): Weight for lane matchup scores.
- `w2` (default: 0.25): Weight for player comfort scores.
- `w3` (default: 0.60): Weight for team composition scores.
- `use_memoization` (default: True): Cache payoff calculations and intermediate game states.
- `skip_api_calls` (default: False): Skip slow external calls for faster computation (composition-only mode).
- `show_progress` (default: True): Display progress bars during calculation.
- `beam_width` (default: 5): Number of top candidates to explore at each level in generic drafts.
- `fast_heuristic` (default: True): Use simplified heuristics for speed in generic drafts.

### Draft Parameters

- `max_depth`: Maximum depth to explore via backward induction (None = full tree).
- `auto_ban_counters` (default: True): Automatically ban counter champions based on matchup winrates.
- `bans_per_pick` (default: 1): Number of bans per pick in generic drafts.

## Algorithm Details

### Subgame Perfect Nash Equilibrium

The SPNE solver uses backward induction to find optimal strategies:

1. **Backward Induction**: Starts from terminal game states and works backwards.
2. **Beam Search** (generic drafts): Explores only top N candidates at each level (reduces complexity).
3. **Alpha-Beta Pruning**: Eliminates branches that cannot improve the solution.
4. **Memoization**: Caches payoff calculations and game states keyed by team compositions.
5. **Role Constraints**: Ensures all roles are filled appropriately for both teams.

### Payoff Calculation

The payoff function combines three components:

```
S = w1(LaneMatchups) + w2(Comfort) + w3(TeamComp)
```

- **Lane Matchups**: Win rate differences in each lane (per-role matchup data).
- **Player Comfort**: Player-specific champion performance from database + fallback scraping.
- **Team Composition**: Synergies, role fulfillment, and strategic elements.



Performance can be improved by:

- Reducing `max_depth` (faster but less optimal).
- Reducing `beam_width` (faster but might miss better strategies).
- Setting `skip_api_calls=True` (faster but less accurate, composition-only).
- Using `fast_heuristic=True` for approximate searches.
- Restricting candidate sets to curated role pools (as in `EXTENDED_CHAMPION_POOL` for the Worlds 2025 games).

## Database Schema

The project uses SQLite databases to store champion and comfort data:

- **Champion tags / roles** (e.g., in `lolchampiontags.db`):
  - Role flags, damage types, engage types, etc.
  - Which champions can play which roles.
  - Composition attributes: wave clear, scaling, peel, etc.
- **Player comfort** (e.g., `worlds2025.db`):
  - Player–champion winrates and usage for specific events (like Worlds 2025).

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request if you add new scenarios (e.g. additional pro matches), improve composition scoring, or integrate new data sources.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Data sources: u.gg, Riot Games API, and event-specific databases.
- Game theory concepts: Subgame Perfect Nash Equilibrium, Backward Induction.

## Future Improvements

- [ ] Add support for full ban phase sequencing in generic drafts.
- [ ] Implement simultaneous game Nash equilibrium.
- [ ] Support for different game modes (ARAM, etc.).
- [ ] Web interface for easier interaction.
- [ ] Machine learning integration for better predictions.
