"""
Test script for Lane Matchup Score calculation.
Demonstrates how to calculate lane matchup scores from match URLs or manual input.
"""

from composition_scorer import CompositionScorer


def test_manual_matchup():
    """Test lane matchup calculation with manually provided teams."""
    print("="*60)
    print("TEST: Manual Lane Matchup Calculation")
    print("="*60)
    
    scorer = CompositionScorer()
    
    # Example teams (order: top, jungle, mid, adc, support)
    blue_team = ['Aatrox', 'Lee Sin', 'Ahri', 'Jinx', 'Thresh']
    red_team = ['Garen', 'Graves', 'Zed', 'Caitlyn', 'Lux']
    
    print(f"\nBlue Team: {', '.join(blue_team)}")
    print(f"Red Team: {', '.join(red_team)}")
    print("\nFetching win rates from u.gg for each matchup...")
    print("(This may take a while as it fetches data from u.gg)\n")
    
    # First, fetch and display all win rates
    role_order = ['top', 'jungle', 'mid', 'adc', 'support']
    print("--- FETCHED WIN RATES ---")
    print(f"{'Role':<10} {'Matchup':<30} {'WRb (Blue vs Red)':<20} {'WRr (Red vs Blue)':<20}")
    print("-" * 80)
    
    matchup_data = []
    for i, role in enumerate(role_order):
        blue_champ = blue_team[i]
        red_champ = red_team[i]
        
        print(f"{role.capitalize():<10} {blue_champ} vs {red_champ:<15}", end="", flush=True)
        
        wr_blue_vs_red = scorer.get_champion_matchup_winrate(blue_champ, red_champ, role)
        if wr_blue_vs_red is not None:
            print(f" {wr_blue_vs_red:.2%} ({wr_blue_vs_red:.4f}){'':<10}", end="", flush=True)
        else:
            print(f" {'N/A':<19}", end="", flush=True)
        
        wr_red_vs_blue = scorer.get_champion_matchup_winrate(red_champ, blue_champ, role)
        if wr_red_vs_blue is not None:
            print(f" {wr_red_vs_blue:.2%} ({wr_red_vs_blue:.4f})")
        else:
            print(f" {'N/A':<19}")
        
        matchup_data.append({
            'role': role,
            'blue_champ': blue_champ,
            'red_champ': red_champ,
            'wr_blue_vs_red': wr_blue_vs_red,
            'wr_red_vs_blue': wr_red_vs_blue
        })
    
    print("-" * 80)
    print("\nNow calculating lane matchup score from the fetched data...\n")
    
    try:
        result = scorer.calculate_matchup_manual(blue_team, red_team, w1=0.35, return_details=True)
        
        print("\n--- LANE MATCHUP CALCULATIONS ---")
        print(f"{'Role':<10} {'Matchup':<25} {'WRb':<12} {'WRr':<12} {'WRb - WRr':<12}")
        print("-" * 75)
        
        total_diff = 0.0
        for detail in result['matchup_details']:
            role = detail['role'].capitalize()
            blue_champ = detail['blue_champ']
            red_champ = detail['red_champ']
            matchup = f"{blue_champ} vs {red_champ}"
            
            if detail['wr_blue_vs_red'] is not None:
                wr_b = f"{detail['wr_blue_vs_red']:.2%}"
            else:
                wr_b = "N/A"
            
            if detail['wr_red_vs_blue'] is not None:
                wr_r = f"{detail['wr_red_vs_blue']:.2%}"
            else:
                wr_r = "N/A"
            
            diff = detail['difference']
            total_diff += diff
            
            print(f"{role:<10} {matchup:<25} {wr_b:<12} {wr_r:<12} {diff:+.4f}")
        
        print("-" * 75)
        print(f"\nSummation across all 5 roles:")
        print(f"  ∑(WRb - WRr) = {total_diff:+.4f}")
        print(f"  Average (1/5∑) = {result['avg_difference']:+.4f}")
        print(f"  Weight factor (w1) = {result['w1']}")
        print(f"  Raw score (w1 * 1/5∑) = {result['w1'] * result['avg_difference']:+.4f}")
        print(f"\nFINAL LANE MATCHUP SCORE (clamped): {result['matchup_score']:+.4f}")
        print(f"  (Range: -0.15 to +0.15)")
        
        print("\nFormula: w1 * 1/5∑(WRb - WRr)")
        print("  where WRb = Blue champion win rate vs Red champion")
        print("  and WRr = Red champion win rate vs Blue champion")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        scorer.close()


def test_url_matchup(match_url: str):
    """Test lane matchup calculation from a match URL."""
    print("="*60)
    print("TEST: Lane Matchup Calculation from URL")
    print("="*60)
    
    scorer = CompositionScorer()
    
    print(f"\nMatch URL: {match_url}")
    print("\nParsing match URL and calculating lane matchup score...")
    print("(This may take a while as it fetches data from u.gg)\n")
    
    try:
        result = scorer.calculate_matchup_from_url(match_url, w1=0.35)
        
        print(f"Blue Team: {', '.join(result['blue_team'])}")
        print(f"Red Team: {', '.join(result['red_team'])}")
        print(f"\nLane Matchup Score: {result['matchup_score']:.4f}")
        print(f"  (Range: -0.15 to +0.15)")
        print(f"\nWeight factor (w1): {result['w1']}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nNote: If automatic parsing fails, you can use calculate_matchup_manual()")
        print("      with manually provided team compositions.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        scorer.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If URL provided as argument
        match_url = sys.argv[1]
        test_url_matchup(match_url)
    else:
        # Run manual test
        print("Running manual matchup test...")
        print("To test with a URL, run: python test_matchup.py <match_url>")
        print()
        
        test_manual_matchup()
        
        print("\n" + "="*60)
        print()
        
