"""
Test script to verify OP.GG scraping functionality.
Tests the _get_player_champion_winrate_from_opgg method directly.
"""

from composition_scorer import CompositionScorer


def test_opgg_scraping():
    """Test OP.GG scraping directly."""
    print("="*70)
    print("TEST: OP.GG Champion Winrate Extraction")
    print("="*70)
    
    scorer = CompositionScorer()
    
    # Use the example from the user's message
    player_name = input("\nEnter player name (e.g., 'DTBOSS-2003' or 'dtboss#2003'): ").strip() or 'DTBOSS-2003'
    champion = input("Enter champion name (e.g., 'Seraphine'): ").strip() or 'Seraphine'
    region = input("Enter region (default: na1): ").strip() or 'na1'
    
    print(f"\nTesting OP.GG scraping for:")
    print(f"  Player: {player_name}")
    print(f"  Champion: {champion}")
    print(f"  Region: {region}")
    
    # Build expected URL
    if '#' in player_name:
        game_name, tag_line = player_name.split('#', 1)
        player_normalized = f"{game_name}-{tag_line}".replace(' ', '-').lower()
    else:
        player_normalized = player_name.replace('#', '-').replace(' ', '-').lower()
    
    opgg_region_map = {
        'na1': 'na', 'euw1': 'euw', 'eun1': 'eune',
        'kr': 'kr', 'jp1': 'jp', 'br1': 'br',
        'la1': 'lan', 'la2': 'las', 'oc1': 'oce',
        'tr1': 'tr', 'ru': 'ru'
    }
    opgg_region = opgg_region_map.get(region.lower(), 'na')
    expected_url = f"https://op.gg/lol/summoners/{opgg_region}/{player_normalized}/champions"
    
    print(f"\nExpected OP.GG URL: {expected_url}")
    print("\n" + "-"*70)
    
    try:
        # Call OP.GG method directly with debug enabled
        stats = scorer._get_player_champion_winrate_from_opgg(
            player_name, champion, region, min_games=10, debug=True
        )
        
        print("\n" + "-"*70)
        
        if stats:
            print(f"\n✓ Successfully extracted data from OP.GG:")
            print(f"  Winrate: {stats['winrate']:.2%} ({stats['winrate']:.4f})")
            print(f"  Games Played: {stats['games_played']}")
            
            if stats['games_played'] >= 10:
                print(f"  ✓ Meets minimum games requirement (≥10)")
            else:
                print(f"  ⚠ Using default 0.5 winrate (< 10 games)")
        else:
            print(f"\n✗ Could not extract winrate data from OP.GG")
            print(f"  Possible reasons:")
            print(f"    - Player name format incorrect")
            print(f"    - Champion name doesn't match OP.GG format")
            print(f"    - Player doesn't have stats for this champion")
            print(f"    - OP.GG page structure changed")
            print(f"    - Network/request error")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


def test_opgg_vs_ugg():
    """Compare OP.GG and u.gg results."""
    print("="*70)
    print("TEST: OP.GG vs u.gg Comparison")
    print("="*70)
    
    scorer = CompositionScorer()
    
    player_name = input("\nEnter player name (e.g., 'DTBOSS-2003'): ").strip() or 'DTBOSS-2003'
    champion = input("Enter champion name: ").strip() or 'Seraphine'
    region = input("Enter region (default: na1): ").strip() or 'na1'
    
    print(f"\nComparing results for {player_name} on {champion}...")
    print("\n" + "-"*70)
    
    try:
        # Test OP.GG
        print("\n[OP.GG] Fetching data...")
        opgg_stats = scorer._get_player_champion_winrate_from_opgg(
            player_name, champion, region, min_games=10, debug=False
        )
        
        # Test u.gg (by disabling Riot API and forcing u.gg)
        print("\n[u.gg] Fetching data...")
        ugg_stats = scorer.get_player_champion_winrate(
            player_name, champion, region, min_games=10, 
            debug=False, use_riot_api=False
        )
        
        print("\n" + "="*70)
        print("RESULTS COMPARISON")
        print("="*70)
        
        if opgg_stats:
            print(f"\n[OP.GG]")
            print(f"  Winrate: {opgg_stats['winrate']:.2%}")
            print(f"  Games: {opgg_stats['games_played']}")
        else:
            print(f"\n[OP.GG] ✗ No data found")
        
        if ugg_stats:
            print(f"\n[u.gg]")
            print(f"  Winrate: {ugg_stats['winrate']:.2%}")
            print(f"  Games: {ugg_stats['games_played']}")
        else:
            print(f"\n[u.gg] ✗ No data found")
        
        if opgg_stats and ugg_stats:
            print(f"\n[Comparison]")
            wr_diff = abs(opgg_stats['winrate'] - ugg_stats['winrate'])
            games_diff = abs(opgg_stats['games_played'] - ugg_stats['games_played'])
            print(f"  Winrate difference: {wr_diff:.4f} ({wr_diff*100:.2f}%)")
            print(f"  Games difference: {games_diff}")
            
            if wr_diff < 0.01 and games_diff <= 1:
                print(f"  ✓ Results match closely!")
            else:
                print(f"  ⚠ Results differ - may need investigation")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


if __name__ == "__main__":
    import sys
    
    print("OP.GG Scraping Test")
    print("\nOptions:")
    print("  1. Test OP.GG scraping only")
    print("  2. Compare OP.GG vs u.gg results")
    
    choice = input("\nEnter choice (1/2): ").strip()
    
    if choice == '1':
        test_opgg_scraping()
    elif choice == '2':
        test_opgg_vs_ugg()
    else:
        print("Invalid choice")

