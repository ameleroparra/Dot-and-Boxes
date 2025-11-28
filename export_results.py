"""
Generate summary statistics and export results to CSV for further analysis
"""

import json
import os
import csv
from collections import defaultdict


def export_game_results():
    """Export game-by-game results to CSV."""
    data_dir = "data/raw"
    results = []
    
    game_dirs = sorted([d for d in os.listdir(data_dir) 
                       if d.startswith("game_") and 
                       os.path.isdir(os.path.join(data_dir, d))],
                      key=lambda x: int(x.split("_")[1]))
    
    for game_dir in game_dirs:
        game_id = int(game_dir.split("_")[1])
        game_path = os.path.join(data_dir, game_dir)
        
        # Get final scores
        turn_files = sorted([f for f in os.listdir(game_path) if f.endswith('.json')])
        final_path = os.path.join(game_path, turn_files[-1])
        
        with open(final_path, 'r') as f:
            final_data = json.load(f)
        
        scores = final_data.get('scores', {})
        red_score = scores.get('red', 0)
        blue_score = scores.get('blue', 0)
        
        winner = 'Bot' if red_score > blue_score else 'VLM' if blue_score > red_score else 'Tie'
        
        results.append({
            'game_id': game_id,
            'vlm_score': blue_score,
            'bot_score': red_score,
            'winner': winner,
            'total_turns': len(turn_files),
            'score_difference': blue_score - red_score,
            'vlm_percentage': blue_score / (red_score + blue_score) * 100 if (red_score + blue_score) > 0 else 0
        })
    
    # Write to CSV
    with open('game_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['game_id', 'vlm_score', 'bot_score', 
                                                'winner', 'total_turns', 'score_difference', 
                                                'vlm_percentage'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Exported {len(results)} game results to game_results.csv")
    return results


def export_move_analysis():
    """Export move-by-move analysis to CSV."""
    data_dir = "data/raw"
    moves = []
    
    game_dirs = sorted([d for d in os.listdir(data_dir) 
                       if d.startswith("game_") and 
                       os.path.isdir(os.path.join(data_dir, d))],
                      key=lambda x: int(x.split("_")[1]))
    
    for game_dir in game_dirs:
        game_id = int(game_dir.split("_")[1])
        game_path = os.path.join(data_dir, game_dir)
        turn_files = sorted([f for f in os.listdir(game_path) if f.endswith('.json')])
        
        for i, turn_file in enumerate(turn_files[1:], 1):
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
                prev_scores = prev_data.get('scores', {})
                curr_scores = turn_data.get('scores', {})
                
                completed_box = (curr_scores.get('red', 0) + curr_scores.get('blue', 0)) > \
                               (prev_scores.get('red', 0) + prev_scores.get('blue', 0))
                
                moves_made = turn_data.get('turn', i)
                if moves_made < 12:
                    phase = 'early'
                elif moves_made < 28:
                    phase = 'mid'
                else:
                    phase = 'late'
                
                moves.append({
                    'game_id': game_id,
                    'turn': i,
                    'player': 'VLM' if player == 1 else 'Bot',
                    'move_type': move[0],
                    'move_i': move[1],
                    'move_j': move[2],
                    'phase': phase,
                    'completed_box': completed_box,
                    'score_before': prev_scores.get('blue' if player == 1 else 'red', 0),
                    'score_after': curr_scores.get('blue' if player == 1 else 'red', 0)
                })
    
    # Write to CSV
    with open('move_analysis.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['game_id', 'turn', 'player', 'move_type', 
                                                'move_i', 'move_j', 'phase', 'completed_box',
                                                'score_before', 'score_after'])
        writer.writeheader()
        writer.writerows(moves)
    
    print(f"Exported {len(moves)} moves to move_analysis.csv")
    return moves


def print_summary_statistics(game_results, moves):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Game-level statistics
    total_games = len(game_results)
    vlm_wins = sum(1 for g in game_results if g['winner'] == 'VLM')
    bot_wins = sum(1 for g in game_results if g['winner'] == 'Bot')
    ties = sum(1 for g in game_results if g['winner'] == 'Tie')
    
    print(f"\nGame Results:")
    print(f"  Total games: {total_games}")
    print(f"  VLM wins: {vlm_wins} ({vlm_wins/total_games*100:.1f}%)")
    print(f"  Bot wins: {bot_wins} ({bot_wins/total_games*100:.1f}%)")
    print(f"  Ties: {ties} ({ties/total_games*100:.1f}%)")
    
    # Score statistics
    vlm_scores = [g['vlm_score'] for g in game_results]
    bot_scores = [g['bot_score'] for g in game_results]
    
    print(f"\nScore Statistics:")
    print(f"  VLM average: {sum(vlm_scores)/len(vlm_scores):.2f}")
    print(f"  VLM min/max: {min(vlm_scores)}/{max(vlm_scores)}")
    print(f"  Bot average: {sum(bot_scores)/len(bot_scores):.2f}")
    print(f"  Bot min/max: {min(bot_scores)}/{max(bot_scores)}")
    
    # Move-level statistics
    vlm_moves = [m for m in moves if m['player'] == 'VLM']
    bot_moves = [m for m in moves if m['player'] == 'Bot']
    
    print(f"\nMove Statistics:")
    print(f"  Total moves: {len(moves)}")
    print(f"  VLM moves: {len(vlm_moves)}")
    print(f"  Bot moves: {len(bot_moves)}")
    
    vlm_completions = sum(1 for m in vlm_moves if m['completed_box'])
    bot_completions = sum(1 for m in bot_moves if m['completed_box'])
    
    print(f"\n  VLM box completions: {vlm_completions} ({vlm_completions/len(vlm_moves)*100:.1f}%)")
    print(f"  Bot box completions: {bot_completions} ({bot_completions/len(bot_moves)*100:.1f}%)")
    
    # Phase statistics
    print(f"\nMove Distribution by Phase:")
    for player_name, player_moves in [('VLM', vlm_moves), ('Bot', bot_moves)]:
        print(f"  {player_name}:")
        early = sum(1 for m in player_moves if m['phase'] == 'early')
        mid = sum(1 for m in player_moves if m['phase'] == 'mid')
        late = sum(1 for m in player_moves if m['phase'] == 'late')
        
        print(f"    Early: {early} ({early/len(player_moves)*100:.1f}%)")
        print(f"    Mid: {mid} ({mid/len(player_moves)*100:.1f}%)")
        print(f"    Late: {late} ({late/len(player_moves)*100:.1f}%)")
    
    # Move type distribution
    print(f"\nMove Type Distribution:")
    for player_name, player_moves in [('VLM', vlm_moves), ('Bot', bot_moves)]:
        h_moves = sum(1 for m in player_moves if m['move_type'] == 'h')
        v_moves = sum(1 for m in player_moves if m['move_type'] == 'v')
        print(f"  {player_name}:")
        print(f"    Horizontal: {h_moves} ({h_moves/len(player_moves)*100:.1f}%)")
        print(f"    Vertical: {v_moves} ({v_moves/len(player_moves)*100:.1f}%)")


def main():
    print("=" * 80)
    print("EXPORTING EVALUATION DATA")
    print("=" * 80)
    
    # Export data
    game_results = export_game_results()
    moves = export_move_analysis()
    
    # Print summary
    print_summary_statistics(game_results, moves)
    
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print("\nFiles created:")
    print("  - game_results.csv: Game-level statistics")
    print("  - move_analysis.csv: Move-by-move data")
    print("  - EVALUATION_REPORT.md: Comprehensive analysis report")


if __name__ == "__main__":
    main()
