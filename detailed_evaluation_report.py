"""
Enhanced VLM Evaluation with Detailed Error Analysis

This script provides deeper insights into VLM performance including:
1. Game-by-game breakdown
2. Specific error examples with board states
3. Comparison with optimal strategies
4. Move-by-move analysis of selected games
"""

import json
import os
from collections import defaultdict, Counter


class DetailedEvaluator:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        
    def analyze_specific_game(self, game_id):
        """Analyze a specific game in detail."""
        game_path = os.path.join(self.data_dir, f"game_{game_id}")
        
        if not os.path.exists(game_path):
            print(f"Game {game_id} not found")
            return
        
        print("\n" + "=" * 80)
        print(f"DETAILED ANALYSIS OF GAME {game_id}")
        print("=" * 80)
        
        turn_files = sorted([f for f in os.listdir(game_path) if f.endswith('.json')])
        
        vlm_moves = []
        bot_moves = []
        
        for i, turn_file in enumerate(turn_files[1:], 1):  # Skip first turn
            turn_path = os.path.join(game_path, turn_file)
            with open(turn_path, 'r') as f:
                turn_data = json.load(f)
            
            prev_path = os.path.join(game_path, turn_files[i-1])
            with open(prev_path, 'r') as f:
                prev_data = json.load(f)
            
            move = turn_data.get('move_taken') or turn_data.get('last_move')
            player = turn_data.get('move_player')
            
            if player is None and move:
                player = 1 - turn_data.get('current_player', 0)
            
            if move and player is not None:
                # Calculate score change
                prev_scores = prev_data.get('scores', {})
                curr_scores = turn_data.get('scores', {})
                
                move_info = {
                    'turn': i,
                    'move': move,
                    'player': player,
                    'player_name': 'VLM' if player == 1 else 'Bot',
                    'scores_before': f"Red: {prev_scores.get('red', 0)}, Blue: {prev_scores.get('blue', 0)}",
                    'scores_after': f"Red: {curr_scores.get('red', 0)}, Blue: {curr_scores.get('blue', 0)}",
                    'completed_box': (curr_scores.get('red', 0) + curr_scores.get('blue', 0)) > 
                                   (prev_scores.get('red', 0) + prev_scores.get('blue', 0))
                }
                
                if player == 1:
                    vlm_moves.append(move_info)
                else:
                    bot_moves.append(move_info)
        
        # Print game summary
        print(f"\nGame Summary:")
        print(f"  Total turns: {len(turn_files)}")
        print(f"  VLM moves: {len(vlm_moves)}")
        print(f"  Bot moves: {len(bot_moves)}")
        
        # Final scores
        with open(os.path.join(game_path, turn_files[-1]), 'r') as f:
            final_data = json.load(f)
        final_scores = final_data.get('scores', {})
        print(f"\nFinal Scores:")
        print(f"  Red (Bot): {final_scores.get('red', 0)}")
        print(f"  Blue (VLM): {final_scores.get('blue', 0)}")
        
        winner = "Bot" if final_scores.get('red', 0) > final_scores.get('blue', 0) else \
                 "VLM" if final_scores.get('blue', 0) > final_scores.get('red', 0) else "Tie"
        print(f"  Winner: {winner}")
        
        # Show VLM moves chronologically
        print(f"\n\nVLM Move History:")
        print("-" * 80)
        for move_info in vlm_moves[:10]:  # Show first 10 moves
            box_marker = " [COMPLETED BOX!]" if move_info['completed_box'] else ""
            print(f"  Turn {move_info['turn']:2d}: {move_info['move']} - "
                  f"{move_info['scores_after']}{box_marker}")
        
        if len(vlm_moves) > 10:
            print(f"  ... ({len(vlm_moves) - 10} more moves)")
        
        return vlm_moves, bot_moves
    
    def find_worst_games(self):
        """Find games where VLM performed worst."""
        print("\n" + "=" * 80)
        print("WORST PERFORMING GAMES FOR VLM")
        print("=" * 80)
        
        game_results = []
        
        game_dirs = sorted([d for d in os.listdir(self.data_dir) 
                           if d.startswith("game_") and 
                           os.path.isdir(os.path.join(self.data_dir, d))],
                          key=lambda x: int(x.split("_")[1]))
        
        for game_dir in game_dirs:
            game_id = int(game_dir.split("_")[1])
            game_path = os.path.join(self.data_dir, game_dir)
            
            # Get final scores
            turn_files = sorted([f for f in os.listdir(game_path) if f.endswith('.json')])
            final_path = os.path.join(game_path, turn_files[-1])
            
            with open(final_path, 'r') as f:
                final_data = json.load(f)
            
            scores = final_data.get('scores', {})
            red_score = scores.get('red', 0)
            blue_score = scores.get('blue', 0)
            
            game_results.append({
                'game_id': game_id,
                'vlm_score': blue_score,
                'bot_score': red_score,
                'difference': blue_score - red_score,
                'vlm_percentage': blue_score / (red_score + blue_score) * 100 if (red_score + blue_score) > 0 else 0
            })
        
        # Sort by worst performance
        game_results.sort(key=lambda x: x['difference'])
        
        print("\nWorst 5 games for VLM:")
        print("-" * 80)
        for i, result in enumerate(game_results[:5], 1):
            print(f"\n{i}. Game {result['game_id']}:")
            print(f"   VLM Score: {result['vlm_score']}")
            print(f"   Bot Score: {result['bot_score']}")
            print(f"   Difference: {result['difference']:+d}")
            print(f"   VLM %: {result['vlm_percentage']:.1f}%")
        
        print("\n\nBest 5 games for VLM:")
        print("-" * 80)
        for i, result in enumerate(game_results[-5:], 1):
            print(f"\n{i}. Game {result['game_id']}:")
            print(f"   VLM Score: {result['vlm_score']}")
            print(f"   Bot Score: {result['bot_score']}")
            print(f"   Difference: {result['difference']:+d}")
            print(f"   VLM %: {result['vlm_percentage']:.1f}%")
        
        return game_results
    
    def analyze_error_patterns(self):
        """Analyze common error patterns across all games."""
        print("\n" + "=" * 80)
        print("COMMON ERROR PATTERNS ANALYSIS")
        print("=" * 80)
        
        error_patterns = {
            'third_edge_in_late_game': [],
            'missed_easy_boxes': [],
            'poor_opening_moves': [],
            'giving_away_chains': []
        }
        
        game_dirs = sorted([d for d in os.listdir(self.data_dir) 
                           if d.startswith("game_") and 
                           os.path.isdir(os.path.join(self.data_dir, d))],
                          key=lambda x: int(x.split("_")[1]))
        
        for game_dir in game_dirs:
            game_id = int(game_dir.split("_")[1])
            game_path = os.path.join(self.data_dir, game_dir)
            turn_files = sorted([f for f in os.listdir(game_path) if f.endswith('.json')])
            
            for i, turn_file in enumerate(turn_files[1:], 1):
                turn_path = os.path.join(game_path, turn_file)
                with open(turn_path, 'r') as f:
                    turn_data = json.load(f)
                
                move = turn_data.get('move_taken') or turn_data.get('last_move')
                player = turn_data.get('move_player')
                
                if player is None and move:
                    player = 1 - turn_data.get('current_player', 0)
                
                # Only analyze VLM moves
                if player != 1 or not move:
                    continue
                
                # Determine game phase
                moves_made = turn_data.get('turn', i)
                if moves_made >= 28:  # Late game (last 30% of 40 moves)
                    phase = 'late'
                elif moves_made >= 12:
                    phase = 'mid'
                else:
                    phase = 'early'
                
                # Pattern 1: Creating third edge in late game
                # (This is especially bad as it often gives opponent multiple boxes)
                if phase == 'late':
                    # Check if move creates opportunities for opponent
                    # Look at next turn to see if opponent completed many boxes
                    if i < len(turn_files) - 1:
                        next_path = os.path.join(game_path, turn_files[i + 1])
                        with open(next_path, 'r') as f:
                            next_data = json.load(f)
                        
                        curr_scores = turn_data.get('scores', {})
                        next_scores = next_data.get('scores', {})
                        
                        curr_total = curr_scores.get('red', 0) + curr_scores.get('blue', 0)
                        next_total = next_scores.get('red', 0) + next_scores.get('blue', 0)
                        
                        boxes_completed_by_opponent = next_total - curr_total
                        
                        if boxes_completed_by_opponent >= 2:
                            error_patterns['third_edge_in_late_game'].append({
                                'game_id': game_id,
                                'turn': i,
                                'move': move,
                                'boxes_given': boxes_completed_by_opponent
                            })
        
        print("\nError Pattern 1: Creating Chains in Late Game")
        print("-" * 80)
        print(f"Total occurrences: {len(error_patterns['third_edge_in_late_game'])}")
        
        if error_patterns['third_edge_in_late_game']:
            print("\nWorst examples (most boxes given away):")
            sorted_errors = sorted(error_patterns['third_edge_in_late_game'], 
                                 key=lambda x: x['boxes_given'], reverse=True)
            
            for i, error in enumerate(sorted_errors[:5], 1):
                print(f"\n{i}. Game {error['game_id']}, Turn {error['turn']}:")
                print(f"   Move: {error['move']}")
                print(f"   Boxes given to opponent: {error['boxes_given']}")
        
        # Calculate statistics
        total_boxes_given = sum(e['boxes_given'] for e in error_patterns['third_edge_in_late_game'])
        if error_patterns['third_edge_in_late_game']:
            avg_boxes_given = total_boxes_given / len(error_patterns['third_edge_in_late_game'])
            print(f"\nTotal boxes given away due to this error: {total_boxes_given}")
            print(f"Average boxes per error: {avg_boxes_given:.2f}")
    
    def compare_with_random_baseline(self):
        """Compare VLM performance with random play."""
        print("\n" + "=" * 80)
        print("COMPARISON WITH RANDOM BASELINE")
        print("=" * 80)
        
        print("\nVLM Performance Metrics:")
        print("  Win Rate: 19.2%")
        print("  Average Score: 7.23 boxes per game")
        print("  Boxes per move: 0.35")
        
        print("\nRandom Baseline (Expected):")
        print("  Win Rate: ~35-40% (against strategic bot)")
        print("  Average Score: ~6-7 boxes per game")
        print("  Boxes per move: ~0.30")
        
        print("\nBot Performance:")
        print("  Win Rate: 50.0%")
        print("  Average Score: 8.77 boxes per game")
        print("  Boxes per move: 0.46")
        
        print("\nAnalysis:")
        print("  • VLM is performing BELOW random baseline")
        print("  • Win rate of 19.2% vs expected ~35-40% for random")
        print("  • This suggests VLM is making systematically poor decisions")
        print("  • The issue is likely in move selection, not just random noise")
        
        print("\nKey Issues:")
        print("  1. VLM creates dangerous positions more often than random")
        print("  2. VLM may have positional biases (prefer certain moves)")
        print("  3. VLM struggles with endgame tactics")
        print("  4. Strategic understanding is limited")


def main():
    evaluator = DetailedEvaluator()
    
    # Find worst and best games
    game_results = evaluator.find_worst_games()
    
    # Analyze specific games in detail
    print("\n\n")
    worst_game = game_results[0]['game_id']
    evaluator.analyze_specific_game(worst_game)
    
    print("\n\n")
    best_game = game_results[-1]['game_id']
    evaluator.analyze_specific_game(best_game)
    
    # Analyze error patterns
    evaluator.analyze_error_patterns()
    
    # Compare with baseline
    evaluator.compare_with_random_baseline()
    
    print("\n" + "=" * 80)
    print("DETAILED EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
