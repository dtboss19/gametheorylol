"""
Test script for League of Legends Team Composition Scorer
Allows users to input 5 champions and see their team composition score.
"""

from composition_scorer import CompositionScorer
import json


def print_score_breakdown(result: dict):
    """Pretty print the composition score breakdown."""
    print("\n" + "="*60)
    print("TEAM COMPOSITION SCORE BREAKDOWN")
    print("="*60)
    
    print(f"\nChampions: {', '.join(result['champions'])}")
    
    print("\n--- STATISTICS ---")
    stats = result['stats']
    print(f"Wave Clear: {stats['wave_clear']}")
    print(f"Frontline/Tank: {stats['frontline_count']} (Tanks: {stats['tank_count']}, Bruiser Front: {stats['bruiser_front_count']})")
    print(f"Bruisers: {stats['bruiser_total']} (Front: {stats['bruiser_front_count']}, Diver: {stats['bruiser_diver_count']})")
    print(f"Damage: AD={stats['ad_count']}, AP={stats['ap_count']}, Mixed={stats['mixed_count']}")
    print(f"Roles: Top={stats['top_count']}, Jungle={stats['jungle_count']}, Mid={stats['mid_count']}, ADC={stats['adc_count']}, Support={stats['support_count']}")
    print(f"Poke: {stats['poke_count']}")
    print(f"Scaling: {stats['scaling_count']}")
    print(f"Pick: {stats['pick_count']}, Diver: {stats['diver_count']}")
    print(f"AOE Combo: {stats['aoe_combo_count']}")
    print(f"Team Fight: {stats['team_fight_count']}")
    print(f"Split Pusher: {stats['split_pusher_count']}")
    print(f"Peel: {stats['peel_count']}, HyperCarry: {stats['hypercarry_count']}")
    print(f"Global: {stats['global_count']}")
    
    print("\n--- BONUSES ---")
    bonuses = result['bonuses']
    for category, value in bonuses.items():
        if value > 0:
            print(f"  {category.replace('_', ' ').title()}: +{value:.3f}")
    
    print("\n--- PENALTIES ---")
    penalties = result['penalties']
    for category, value in penalties.items():
        if value > 0:
            print(f"  {category.replace('_', ' ').title()}: -{value:.3f}")
    
    print("\n--- FINAL SCORE ---")
    print(f"Total Bonus: +{result['total_bonus']:.3f}")
    print(f"Total Penalty: -{result['total_penalty']:.3f}")
    print(f"Raw Score: {result.get('raw_score', result['total_bonus'] - result['total_penalty']):.3f}")
    if 'normalized_score' in result:
        print(f"Normalized Score: {result['normalized_score']:.3f}")
        print(f"Weight factor (w3): {result.get('w3', 0.3)}")
        print(f"COMPOSITION SCORE (w3 * Normalized): {result['total_score']:.3f}")
    else:
        print(f"COMPOSITION SCORE (Normalized): {result['total_score']:.3f}")
    print(f"  (Range: -0.15 to +0.15)")
    print("="*60 + "\n")


def interactive_mode():
    """Interactive mode for user input."""
    scorer = CompositionScorer()
    
    print("League of Legends Team Composition Scorer")
    print("Enter 5 champion names (one per line):")
    print("(Type 'quit' to exit, 'example' for a sample team)\n")
    
    while True:
        champions = []
        
        for i in range(5):
            champ = input(f"Champion {i+1}/5: ").strip()
            
            if champ.lower() == 'quit':
                scorer.close()
                return
            
            if champ.lower() == 'example':
                # Example team
                champions = ['Aatrox', 'Ahri', 'Jinx', 'Thresh', 'Lee Sin']
                print(f"\nUsing example team: {', '.join(champions)}")
                break
            
            if not champ:
                print("Please enter a champion name.")
                i -= 1
                continue
            
            champions.append(champ)
        
        if len(champions) == 5:
            try:
                result = scorer.calculate_composition_score(champions)
                print_score_breakdown(result)
            except ValueError as e:
                print(f"\nError: {e}\n")
        
        print("\nEnter another team? (y/n): ", end='')
        if input().strip().lower() != 'y':
            break
        print()
    
    scorer.close()


def batch_mode(champion_lists: list):
    """Test multiple teams at once."""
    scorer = CompositionScorer()
    
    for i, champions in enumerate(champion_lists, 1):
        print(f"\n{'='*60}")
        print(f"TEAM {i}")
        print('='*60)
        try:
            result = scorer.calculate_composition_score(champions)
            print_score_breakdown(result)
        except ValueError as e:
            print(f"Error: {e}\n")
    
    scorer.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode: python test_composition.py "Champ1,Champ2,Champ3,Champ4,Champ5"
        champions = [c.strip() for c in sys.argv[1].split(',')]
        if len(champions) != 5:
            print("Error: Must provide exactly 5 champions")
            sys.exit(1)
        
        scorer = CompositionScorer()
        try:
            result = scorer.calculate_composition_score(champions)
            print_score_breakdown(result)
        except ValueError as e:
            print(f"Error: {e}")
        finally:
            scorer.close()
    else:
        # Interactive mode
        interactive_mode()

