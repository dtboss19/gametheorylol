"""
Test script for Riot Games API integration.
Demonstrates how to get PUUID, match history, and match data.
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


def test_get_puuid():
    """Test getting PUUID from Riot ID."""
    print("="*70)
    print("TEST: Get PUUID from Riot ID")
    print("="*70)
    
    api_key = os.getenv('RIOT_API_KEY')
    if not api_key:
        print("\nError: Set RIOT_API_KEY environment variable")
        print("Example: export RIOT_API_KEY='your_key_here'")
        return
    
    scorer = CompositionScorer(riot_api_key=api_key)
    
    game_name = input("\nEnter game name (e.g., 'dtboss'): ").strip()
    tag_line = input("Enter tag line (e.g., '2003'): ").strip()
    region = input("Enter region (americas/europe/asia, default: americas): ").strip() or 'americas'
    
    try:
        puuid = scorer.get_puuid_from_riot_id(game_name, tag_line, region)
        print(f"\nPUUID: {puuid}")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        scorer.close()


def test_get_match_history():
    """Test getting match history from Riot ID."""
    print("="*70)
    print("TEST: Get Match History from Riot ID")
    print("="*70)
    
    api_key = os.getenv('RIOT_API_KEY')
    if not api_key:
        print("\nError: Set RIOT_API_KEY environment variable")
        print("Example: export RIOT_API_KEY='your_key_here'")
        return
    
    scorer = CompositionScorer(riot_api_key=api_key)
    
    game_name = input("\nEnter game name (e.g., 'dtboss'): ").strip()
    tag_line = input("Enter tag line (e.g., '2003'): ").strip()
    region = input("Enter region (americas/europe/asia, default: americas): ").strip() or 'americas'
    count = input("Number of matches to fetch (default: 20, max: 100): ").strip()
    count = int(count) if count else 20
    
    try:
        match_ids = scorer.get_match_history_by_riot_id(game_name, tag_line, region, count=count)
        print(f"\nFound {len(match_ids)} matches:")
        for i, match_id in enumerate(match_ids[:10], 1):  # Show first 10
            print(f"  {i}. {match_id}")
        if len(match_ids) > 10:
            print(f"  ... and {len(match_ids) - 10} more")
        
        if match_ids:
            print(f"\nFirst match ID: {match_ids[0]}")
            print("You can use this with test_complete_match.py:")
            print(f"  python test_complete_match.py riot {match_ids[0]} na1")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


def test_get_match_from_history():
    """Test getting a match from match history and calculating scores."""
    print("="*70)
    print("TEST: Get Match from History and Calculate Scores")
    print("="*70)
    
    api_key = os.getenv('RIOT_API_KEY')
    if not api_key:
        print("\nError: Set RIOT_API_KEY environment variable")
        print("Example: export RIOT_API_KEY='your_key_here'")
        return
    
    scorer = CompositionScorer(riot_api_key=api_key)
    
    game_name = input("\nEnter game name (e.g., 'dtboss'): ").strip()
    tag_line = input("Enter tag line (e.g., '2003'): ").strip()
    region = input("Enter region (americas/europe/asia, default: americas): ").strip() or 'americas'
    
    try:
        # Get match history
        print("\nFetching match history...")
        match_ids = scorer.get_match_history_by_riot_id(game_name, tag_line, region, count=5)
        
        if not match_ids:
            print("No matches found")
            return
        
        print(f"\nFound {len(match_ids)} matches. Using most recent match...")
        match_id = match_ids[0]
        print(f"Match ID: {match_id}")
        
        # Calculate complete match score
        print("\nCalculating match scores...")
        result = scorer.calculate_complete_match_score(
            match_id=match_id,
            region='na1',  # Platform region, not API region
            use_riot_api=True,
            w1=0.35, w2=0.35, w3=0.3
        )
        
        # Display results
        print("\n" + "="*70)
        print("MATCH RESULTS")
        print("="*70)
        print(f"\nBlue Team: {', '.join(result['blue_team'])}")
        print(f"Red Team: {', '.join(result['red_team'])}")
        print(f"\nS Score: {result.get('S_score', 0.0):+.4f}")
        print(f"Blue Win Probability: {result.get('U_blue', 0.5):.2%}")
        print(f"Red Win Probability: {result.get('red_win_probability', 0.5):.2%}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scorer.close()


if __name__ == "__main__":
    import sys
    
    print("Riot Games API Test Script")
    print("\nOptions:")
    print("  1. Get PUUID from Riot ID")
    print("  2. Get Match History from Riot ID")
    print("  3. Get Match and Calculate Scores")
    print("\nMake sure RIOT_API_KEY environment variable is set!")
    print("Example: export RIOT_API_KEY='your_key_here'")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        test_get_puuid()
    elif choice == '2':
        test_get_match_history()
    elif choice == '3':
        test_get_match_from_history()
    else:
        print("Invalid choice")

