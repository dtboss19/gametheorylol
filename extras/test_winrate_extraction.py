"""
Test script to verify player champion winrate extraction from u.gg.
Helps debug winrate fetching issues.
"""

from composition_scorer import CompositionScorer
import os

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def test_player_winrate():
    """Test fetching a single player's winrate on a champion."""
    print("="*70)
    print("TEST: Player Champion Winrate Extraction")
    print("="*70)
    
    scorer = CompositionScorer()
    
    player_name = input("\nEnter player name (e.g., 'dtboss#2003' or 'dtboss-2003'): ").strip()
    champion = input("Enter champion name: ").strip()
    region = input("Enter region (default: na1): ").strip() or 'na1'
    
    print(f"\nFetching winrate for {player_name} on {champion}...")
    print(f"URL: https://u.gg/lol/profile/{region}/{player_name.replace('#', '-').lower()}/champion-stats")
    
    try:
        stats = scorer.get_player_champion_winrate(player_name, champion, region, min_games=10)
        
        if stats:
            print(f"\n✓ Successfully extracted data:")
            print(f"  Winrate: {stats['winrate']:.2%} ({stats['winrate']:.4f})")
            print(f"  Games Played: {stats['games_played']}")
            
            if stats['games_played'] >= 10:
                print(f"  ✓ Meets minimum games requirement (≥10)")
            else:
                print(f"  ⚠ Using default 0.5 winrate (< 10 games)")
        else:
            print(f"\n✗ Could not extract winrate data")
            print(f"  Possible reasons:")
            print(f"    - Player name format incorrect")
            print(f"    - Champion name doesn't match u.gg format")
            print(f"    - Player doesn't have stats for this champion")
            print(f"    - u.gg page structure changed")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


def test_match_players():
    """Test fetching winrates for all players in a match."""
    print("="*70)
    print("TEST: Winrate Extraction for Match Players")
    print("="*70)
    
    api_key = os.getenv('RIOT_API_KEY')
    if not api_key:
        print("\nError: Set RIOT_API_KEY in .env file")
        return
    
    scorer = CompositionScorer(riot_api_key=api_key)
    
    match_id = input("\nEnter Riot match ID (e.g., 'NA1_1234567890'): ").strip()
    region = input("Enter API region (default: americas): ").strip() or 'americas'
    
    try:
        # Fetch match data
        print("\nFetching match data...")
        match_data = scorer.fetch_match_by_riot_id(match_id, region)
        
        blue_team = match_data['blue_team']
        red_team = match_data['red_team']
        blue_players = match_data.get('blue_players', [])
        red_players = match_data.get('red_players', [])
        
        print(f"\nBlue Team: {', '.join(blue_team)}")
        print(f"Blue Players: {', '.join(blue_players)}")
        print(f"\nRed Team: {', '.join(red_team)}")
        print(f"Red Players: {', '.join(red_players)}")
        
        if not blue_players or not red_players:
            print("\n⚠ Player names not found in match data")
            return
        
        # Test winrate extraction for each player
        print("\n" + "="*70)
        print("Testing Winrate Extraction")
        print("="*70)
        
        role_order = ['Top', 'Jungle', 'Mid', 'ADC', 'Support']
        platform_region = 'na1'  # You may need to adjust this
        
        for i, role in enumerate(role_order):
            blue_player = blue_players[i]
            red_player = red_players[i]
            blue_champ = blue_team[i]
            red_champ = red_team[i]
            
            print(f"\n{role}:")
            print(f"  Blue: {blue_player} on {blue_champ}")
            blue_stats = scorer.get_player_champion_winrate(blue_player, blue_champ, platform_region, min_games=10)
            if blue_stats:
                print(f"    ✓ WR: {blue_stats['winrate']:.2%}, Games: {blue_stats['games_played']}")
            else:
                print(f"    ✗ Could not fetch winrate")
            
            print(f"  Red: {red_player} on {red_champ}")
            red_stats = scorer.get_player_champion_winrate(red_player, red_champ, platform_region, min_games=10)
            if red_stats:
                print(f"    ✓ WR: {red_stats['winrate']:.2%}, Games: {red_stats['games_played']}")
            else:
                print(f"    ✗ Could not fetch winrate")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


if __name__ == "__main__":
    import sys
    
    print("Winrate Extraction Test")
    print("\nOptions:")
    print("  1. Test single player winrate")
    print("  2. Test all players in a match")
    
    choice = input("\nEnter choice (1/2): ").strip()
    
    if choice == '1':
        test_player_winrate()
    elif choice == '2':
        test_match_players()
    else:
        print("Invalid choice")

