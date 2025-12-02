# League of Legends Game Theory Solver

A Python implementation of Subgame Perfect Nash Equilibrium (SPNE) solver for League of Legends draft phase analysis. This project uses backward induction to find optimal champion selection strategies based on lane matchups, player comfort, and team composition.

## Features

- **Subgame Perfect Nash Equilibrium (SPNE) Calculation**: Uses backward induction to find optimal draft strategies
- **Team Composition Scoring**: Evaluates team compositions based on synergies, role requirements, and strategic elements
- **Lane Matchup Analysis**: Calculates win rates between champions in specific roles
- **Player Comfort Integration**: Incorporates player-specific champion performance data
- **Automatic Counter-Banning**: Automatically bans champions with high win rates against picked champions
- **Role-Based Champion Selection**: Ensures teams fulfill all required roles (Top, Jungle, Mid, ADC, Support)
- **Optimized Performance**: Uses beam search, memoization, and alpha-beta pruning for efficient computation

## Project Structure

```
gametheorylol/
├── nash_equilibrium.py      # SPNE solver implementation
├── composition_scorer.py    # Team composition and matchup scoring
├── benchmark_spne.py        # Performance benchmarking tools
├── import_roles.py          # Database import utilities
├── test_nash_equilibrium.py # Main test script
├── test_complete_match.py   # Complete match analysis tests
├── test_composition.py      # Composition scoring tests
├── test_matchup.py          # Matchup analysis tests
├── test_player_comfort.py   # Player comfort tests
├── test_riot_api.py         # Riot API integration tests
├── test_opgg_scraping.py    # OP.GG scraping tests
├── test_winrate_extraction.py # Winrate extraction tests
├── requirements.txt         # Python dependenciesg
├── lolchampiontags.db      # Main champion database
└── README.md               # This file
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

3. Set up the database:
   - Ensure `lolchampiontags.db` exists in the project root
   - Run `import_roles.py` if you need to import role data

4. (Optional) Set up Riot API key for player comfort data:
   - Create a `.env` file
   - Add: `RIOT_API_KEY=your_api_key_here`

## Usage

### Basic SPNE Calculation

```python
from nash_equilibrium import SubgamePerfectNashEquilibrium
from composition_scorer import CompositionScorer

# Initialize
scorer = CompositionScorer()
spne = SubgamePerfectNashEquilibrium(
    scorer,
    w1=0.35,  # Weight for lane matchups
    w2=0.35,  # Weight for player comfort
    w3=0.30,  # Weight for composition score
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

### Running Tests

```bash
# Run main SPNE test
python test_nash_equilibrium.py

# Run specific test
python test_nash_equilibrium.py sequential

# Run composition tests
python test_composition.py

# Run complete match analysis
python test_complete_match.py
```

### Benchmarking Performance

```bash
python benchmark_spne.py
```

## Configuration Options

### SPNE Solver Parameters

- `w1` (default: 0.35): Weight for lane matchup scores
- `w2` (default: 0.35): Weight for player comfort scores
- `w3` (default: 0.30): Weight for team composition scores
- `use_memoization` (default: True): Cache payoff calculations
- `skip_api_calls` (default: False): Skip slow API calls for faster computation
- `show_progress` (default: True): Display progress during calculation
- `beam_width` (default: 5): Number of top candidates to explore at each level
- `fast_heuristic` (default: True): Use simplified heuristics for speed

### Draft Parameters

- `max_depth`: Maximum depth to explore via backward induction (None = full tree)
- `auto_ban_counters` (default: True): Automatically ban counter champions
- `bans_per_pick` (default: 1): Number of bans per pick

## Algorithm Details

### Subgame Perfect Nash Equilibrium

The SPNE solver uses backward induction to find optimal strategies:

1. **Backward Induction**: Starts from terminal game states and works backwards
2. **Beam Search**: Explores only top N candidates at each level (reduces complexity)
3. **Alpha-Beta Pruning**: Eliminates branches that cannot improve the solution
4. **Memoization**: Caches payoff calculations and game states
5. **Role Constraints**: Ensures all roles are filled appropriately

### Payoff Calculation

The payoff function combines three components:

```
S = w1(LaneMatchups) + w2(Comfort) + w3(TeamComp)
```

- **Lane Matchups**: Win rate differences in each lane
- **Player Comfort**: Player-specific champion performance
- **Team Composition**: Synergies, role fulfillment, and strategic elements

### Auto-Banning

When enabled, the algorithm automatically bans champions with high win rates (≥52%) against picked champions. This simulates strategic counter-banning behavior.

## Performance

- **Small subset (20-30 champions)**: 
- **Medium subset (40-50 champions)**: 
- **Large subset (100+ champions)**: 

Performance can be improved by:
- Reducing `max_depth` (faster but less optimal)
- Reducing `beam_width` (faster but might miss better strategies)
- Setting `skip_api_calls=True` (faster but less accurate)
- Using `fast_heuristic=True` (faster but less accurate)

## Database Schema

The project uses SQLite databases to store champion data:

- **Champion tags**: Role flags, damage types, engage types, etc.
- **Role data**: Which champions can play which roles
- **Composition attributes**: Wave clear, scaling, peel, etc.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Data sources: u.gg, Riot Games API
- Game theory concepts: Subgame Perfect Nash Equilibrium, Backward Induction

## Future Improvements

- [ ] Add support for actual ban phase in draft order
- [ ] Implement simultaneous game Nash equilibrium
- [ ] Add more sophisticated composition scoring
- [ ] Support for different game modes (ARAM, etc.)
- [ ] Web interface for easier interaction
- [ ] Machine learning integration for better predictions

