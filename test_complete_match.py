"""
Test script for Complete Match Score calculation.
Tests all three components (lane matchups, player comfort, composition).

Supports:
1. Riot Games API match ID (requires API key)
2. Manual input of teams and players
"""

from composition_scorer import CompositionScorer
import os

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use environment variables directly
    pass


def test_complete_match_riot_api(match_id: str, region: str = 'na1', api_key: str = None):
    """Test complete match score calculation using Riot Games API."""
    print("="*70)
    print("COMPLETE MATCH SCORE CALCULATION (Riot Games API)")
    print("="*70)
    
    # Get API key from parameter or environment variable
    if not api_key:
        api_key = os.getenv('RIOT_API_KEY')
    
    if not api_key:
        print("\nError: Riot API key required!")
        print("Set it as an environment variable: export RIOT_API_KEY='your_key'")
        print("Or pass it as a parameter: test_complete_match_riot_api(match_id, region, api_key)")
        return
    
    scorer = CompositionScorer(riot_api_key=api_key)
    
    print(f"\nMatch ID: {match_id}")
    print(f"Region: {region}")
    print("\nFetching match data from Riot Games API and calculating all scores...")
    print("(This may take a while as it fetches data)\n")
    
    try:
        result = scorer.calculate_complete_match_score(
            match_id=match_id, region=region,
            w1=0.35, w2=0.35, w3=0.3,
            use_riot_api=True
        )
        _display_results(result)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


def test_complete_match_manual(blue_team: list, red_team: list,
                               blue_players: list = None, red_players: list = None,
                               region: str = 'na1'):
    """Test complete match score calculation with manual input."""
    print("="*70)
    print("COMPLETE MATCH SCORE CALCULATION (Manual Input)")
    print("="*70)
    
    scorer = CompositionScorer()
    
    print(f"\nBlue Team: {', '.join(blue_team)}")
    print(f"Red Team: {', '.join(red_team)}")
    if blue_players:
        print(f"Blue Players: {', '.join(blue_players)}")
    if red_players:
        print(f"Red Players: {', '.join(red_players)}")
    print(f"Region: {region}")
    print("\nCalculating all scores...")
    print("(This may take a while as it fetches data from u.gg)\n")
    
    try:
        result = scorer.calculate_complete_match_score(
            match_id=None, region=region,
            w1=0.35, w2=0.35, w3=0.3,
            blue_team=blue_team, red_team=red_team,
            blue_players=blue_players, red_players=red_players
        )
        _display_results(result)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


def _display_results(result: dict):
    """Display the results of the match score calculation."""
    # Display match info
    print("="*70)
    print("MATCH INFORMATION")
    print("="*70)
    print(f"\nBlue Team Champions: {', '.join(result['blue_team'])}")
    print(f"Red Team Champions: {', '.join(result['red_team'])}")
    
    if result.get('blue_players') and result.get('red_players'):
        print(f"\nBlue Team Players: {', '.join(result['blue_players'])}")
        print(f"Red Team Players: {', '.join(result['red_players'])}")
    else:
        print("\nNote: Player names not extracted from match data")
    
    # Lane Matchup Score
    print("\n" + "="*70)
    print("1. LANE MATCHUP SCORE (w1 = 0.35)")
    print("="*70)
    matchup = result['lane_matchup']
    print(f"\nSum of differences: {matchup['sum_differences']:+.4f}")
    print(f"Average (1/5∑): {matchup['avg_difference']:+.4f}")
    print(f"Raw score (w1 * 1/5∑): {matchup['w1'] * matchup['avg_difference']:+.4f}")
    print(f"LANE MATCHUP SCORE: {matchup['score']:+.4f}")
    
    # Player Comfort Score
    print("\n" + "="*70)
    print("2. PLAYER COMFORT SCORE (w2 = 0.35)")
    print("="*70)
    comfort = result['player_comfort']
    if 'note' in comfort:
        print(f"\n{comfort['note']}")
        print(f"PLAYER COMFORT SCORE: {comfort['score']:+.4f}")
    else:
        print(f"\nSum of differences: {comfort['sum_differences']:+.4f}")
        print(f"Raw score (w2 * ∑): {comfort['raw_score']:+.4f}")
        print(f"PLAYER COMFORT SCORE: {comfort['score']:+.4f}")
        
        # Show breakdown
        print(f"\n{'Role':<10} {'Blue Player':<25} {'Blue WR':<12} {'Games':<8} {'Red Player':<25} {'Red WR':<12} {'Games':<8} {'Diff':<10}")
        print("-" * 110)
        for detail in comfort['comfort_details']:
            role = detail['role'].capitalize()
            blue_p = f"{detail['blue_player']} ({detail['blue_champ']})"
            red_p = f"{detail['red_player']} ({detail['red_champ']})"
            blue_wr = f"{detail['blue_wr']:.2%}" if detail['blue_games'] >= 10 else f"{detail['blue_wr']:.2%}*"
            red_wr = f"{detail['red_wr']:.2%}" if detail['red_games'] >= 10 else f"{detail['red_wr']:.2%}*"
            blue_games = detail['blue_games'] if detail['blue_games'] > 0 else "N/A"
            red_games = detail['red_games'] if detail['red_games'] > 0 else "N/A"
            diff = detail['difference']
            print(f"{role:<10} {blue_p:<25} {blue_wr:<12} {blue_games:<8} {red_p:<25} {red_wr:<12} {red_games:<8} {diff:+.4f}")
        print("* = Using default 0.5 winrate (< 10 games or data not found)")
    
    # Composition Score
    print("\n" + "="*70)
    print("3. COMPOSITION SCORE (w3 = 0.3)")
    print("="*70)
    comp = result['composition']
    if 'error' in comp:
        print(f"\nError calculating composition: {comp['error']}")
    else:
        print(f"\nBlue Team Composition Score: {comp['blue_score']:+.4f}")
        print(f"Red Team Composition Score: {comp['red_score']:+.4f}")
        print(f"Composition Advantage (Blue - Red): {comp['advantage']:+.4f}")
    
    # S Score and Win Probability
    print("\n" + "="*70)
    print("S SCORE AND WIN PROBABILITY")
    print("="*70)
    print(f"\nS = w1(LaneMatchups) + w2(Comfort) + w3(TeamComp)")
    print(f"  = {matchup['score']:+.4f} + {result['player_comfort'].get('score', 0.0):+.4f} + {comp.get('advantage', 0.0):+.4f}")
    print("-" * 70)
    print(f"S Score: {result.get('S_score', 0.0):+.4f}")
    print(f"Alpha (α): {result.get('alpha', 0.02):.4f}")
    print(f"\nU(Blue) = 1 / (1 + e^-(S + α))")
    print(f"        = 1 / (1 + e^-({result.get('S_score', 0.0):+.4f} + {result.get('alpha', 0.02):.4f}))")
    print("-" * 70)
    print(f"Blue Team Win Probability:  {result.get('U_blue', 0.5):.2%}")
    print(f"Red Team Win Probability:   {result.get('red_win_probability', 0.5):.2%}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("S Score: Sum of all three weighted components")
    print("U(Blue): Blue team's win probability using sigmoid function")
    print("  - > 50% = Blue team favored")
    print("  - < 50% = Red team favored")
    print("  - = 50% = Even match")


if __name__ == "__main__":
    import sys
    
    # Get API key from environment variable
    api_key = os.getenv('RIOT_API_KEY')
    
    if not api_key:
        print("\nError: Riot API key required!")
        print("Set it as an environment variable: export RIOT_API_KEY='your_key'")
        print("Or create a .env file with: RIOT_API_KEY=your_key")
        sys.exit(1)
    
    # Parse arguments - skip "riot" if present (for backwards compatibility)
    args = [arg for arg in sys.argv[1:] if arg.lower() != 'riot']
    
    # Import match_id
    if len(args) > 0:
        match_id = args[0]
    else:
        match_id = input("\nEnter match ID (e.g., NA1_1234567890): ").strip()
    
    if not match_id:
        print("Error: Match ID is required")
        sys.exit(1)
    
    # Optional region parameter
    region = args[1] if len(args) > 1 else 'na1'
    
    # Run the test
    test_complete_match_riot_api(match_id, region, api_key)

