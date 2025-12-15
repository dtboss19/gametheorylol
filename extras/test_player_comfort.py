"""
Test script for Player Comfort Score calculation.
Demonstrates how to calculate player comfort scores from player names and champions.
"""

from composition_scorer import CompositionScorer


def test_player_comfort():
    """Test player comfort calculation with player names and champions."""
    print("="*60)
    print("TEST: Player Comfort Score Calculation")
    print("="*60)
    
    scorer = CompositionScorer()
    
    # Example teams with player names and champions
    # Order: [top, jungle, mid, adc, support]
    blue_players = ['Player1', 'Player2', 'Player3', 'Player4', 'Player5']
    red_players = ['Enemy1', 'Enemy2', 'Enemy3', 'Enemy4', 'Enemy5']
    
    blue_team = ['Aatrox', 'Lee Sin', 'Ahri', 'Jinx', 'Thresh']
    red_team = ['Garen', 'Graves', 'Zed', 'Caitlyn', 'Lux']
    
    region = 'na1'  # Change to your region
    
    print(f"\nBlue Team Players: {', '.join(blue_players)}")
    print(f"Blue Team Champions: {', '.join(blue_team)}")
    print(f"\nRed Team Players: {', '.join(red_players)}")
    print(f"Red Team Champions: {', '.join(red_team)}")
    print(f"\nRegion: {region}")
    print("\nFetching player winrates from u.gg...")
    print("(This may take a while as it fetches data for each player)\n")
    
    try:
        result = scorer.calculate_player_comfort_score(
            blue_players, red_players,
            blue_team, red_team,
            region=region, w2=0.35, return_details=True
        )
        
        print("\n--- PLAYER COMFORT BREAKDOWN (All 5 Roles) ---")
        print(f"{'Role':<10} {'Blue Player':<20} {'Blue WR':<12} {'Red Player':<20} {'Red WR':<12} {'Difference':<12}")
        print("-" * 90)
        
        total_diff = 0.0
        for detail in result['comfort_details']:
            role = detail['role'].capitalize()
            blue_player = detail['blue_player']
            blue_champ = detail['blue_champ']
            blue_wr = detail['blue_wr']
            blue_games = detail['blue_games']
            
            red_player = detail['red_player']
            red_champ = detail['red_champ']
            red_wr = detail['red_wr']
            red_games = detail['red_games']
            
            diff = detail['difference']
            total_diff += diff
            
            blue_display = f"{blue_player} ({blue_champ})"
            red_display = f"{red_player} ({red_champ})"
            
            blue_wr_str = f"{blue_wr:.2%}" if blue_games >= 10 else f"{blue_wr:.2%}*"
            red_wr_str = f"{red_wr:.2%}" if red_games >= 10 else f"{red_wr:.2%}*"
            
            print(f"{role:<10} {blue_display:<20} {blue_wr_str:<12} {red_display:<20} {red_wr_str:<12} {diff:+.4f}")
        
        print("-" * 90)
        print("* = Using default 0.5 winrate (player has < 10 games on champion)")
        print(f"\nSummation across all 5 roles:")
        print(f"  ∑(Blue Player WR - Red Player WR) = {total_diff:+.4f}")
        print(f"  Weight factor (w2) = {result['w2']}")
        print(f"  Raw score (w2 * ∑) = {result['raw_score']:+.4f}")
        print(f"\nFINAL PLAYER COMFORT SCORE (clamped): {result['score']:+.4f}")
        print(f"  (Range: -0.15 to +0.15)")
        
        print("\nFormula: w2 * ∑(Player WR on champ - Enemy WR on champ)")
        print("  where Player WR = Player's winrate on their champion (0.5 if < 10 games)")
        print("  and Enemy WR = Enemy player's winrate on their champion (0.5 if < 10 games)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


if __name__ == "__main__":
    test_player_comfort()

