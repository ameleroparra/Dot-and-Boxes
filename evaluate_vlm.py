"""
VLM Evaluation Script for Dots and Boxes

This script evaluates the VLM's performance by:
1. Analyzing all game data to identify VLM moves
2. Computing quantitative metrics (win rate, completion rate, move quality)
3. Performing qualitative analysis of move patterns and errors
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import random


class DotsAndBoxesEvaluator:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        self.games = []
        self.vlm_moves = []
        self.bot_moves = []
        
    def load_all_games(self):
        """Load all game data from JSON files."""
        print("Loading game data...")
        
        game_dirs = sorted([d for d in os.listdir(self.data_dir) 
                           if d.startswith("game_") and 
                           os.path.isdir(os.path.join(self.data_dir, d))],
                          key=lambda x: int(x.split("_")[1]))
        
        for game_dir in game_dirs:
            game_path = os.path.join(self.data_dir, game_dir)
            game_data = self.load_game(game_path)
            if game_data:
                self.games.append(game_data)
        
        print(f"Loaded {len(self.games)} games")
        return self.games
    
    def load_game(self, game_path):
        """Load all turns for a single game."""
        turn_files = sorted([f for f in os.listdir(game_path) if f.endswith('.json')])
        
        game_data = {
            'game_id': None,
            'turns': [],
            'final_result': None
        }
        
        for turn_file in turn_files:
            turn_path = os.path.join(game_path, turn_file)
            try:
                with open(turn_path, 'r') as f:
                    turn_data = json.load(f)
                    game_data['turns'].append(turn_data)
                    
                    if game_data['game_id'] is None:
                        game_data['game_id'] = turn_data.get('game_id')
                    
                    # Check if game completed (new format)
                    if turn_data.get('result', {}).get('game_completed'):
                        game_data['final_result'] = turn_data['result']
            except Exception as e:
                print(f"Error loading {turn_path}: {e}")
                continue
        
        # For old format games, infer completion from last turn
        if game_data['turns'] and game_data['final_result'] is None:
            last_turn = game_data['turns'][-1]
            remaining = last_turn.get('remaining_lines', last_turn.get('move_info', {}).get('remaining_moves', -1))
            
            if remaining == 0 or len(last_turn.get('available_moves', [])) == 0:
                # Game is complete, reconstruct result
                scores = last_turn.get('scores', {})
                red_score = scores.get('red', 0)
                blue_score = scores.get('blue', 0)
                
                game_data['final_result'] = {
                    'game_completed': True,
                    'final_scores': {'red': red_score, 'blue': blue_score},
                    'winner': 0 if red_score > blue_score else 1 if blue_score > red_score else 'tie'
                }
        
        return game_data if game_data['turns'] else None
    
    def identify_player_roles(self, game):
        """Determine which player is VLM and which is Bot based on game patterns."""
        # In "Bot vs VLM" mode: Player 0 (red) is Bot, Player 1 (blue) is VLM
        # We can infer this from the game structure
        
        # Simple heuristic: VLM is player 1 in most recorded games
        return {'vlm_player': 1, 'bot_player': 0}
    
    def analyze_move_quality(self, turn_data, move, prev_turn_data=None):
        """Analyze the quality of a move based on game state."""
        if not move:
            return None
        
        move_type, i, j = move
        
        # Determine game phase from move count
        moves_made = turn_data.get('move_info', {}).get('moves_made', 
                                                         turn_data.get('turn', 0))
        total_moves = 40  # 4x4 grid has 40 possible lines
        
        if moves_made < total_moves * 0.3:
            phase = 'early'
        elif moves_made < total_moves * 0.7:
            phase = 'mid'
        else:
            phase = 'late'
        
        # Check if box was completed by comparing scores
        completed_box = False
        if prev_turn_data:
            prev_scores = prev_turn_data.get('scores', {})
            curr_scores = turn_data.get('scores', {})
            prev_total = prev_scores.get('red', 0) + prev_scores.get('blue', 0)
            curr_total = curr_scores.get('red', 0) + curr_scores.get('blue', 0)
            completed_box = curr_total > prev_total
        
        analysis = {
            'move': move,
            'game_phase': turn_data.get('strategy_info', {}).get('game_phase', phase),
            'completed_box': turn_data.get('strategy_info', {}).get('completed_boxes_this_turn', completed_box),
            'creates_third_edge': False,
            'safe_move': False,
            'available_moves_count': len(turn_data.get('move_info', {}).get('available_moves', 
                                                                            turn_data.get('available_moves', []))),
        }
        
        # Check if move creates third edge (potentially dangerous)
        # Use new format if available, otherwise old format
        board_state = turn_data.get('board_state')
        if not board_state:
            # Convert old format to new format
            board_state = {
                'grid_size': 4,
                'horizontal_lines': turn_data.get('horizontal_lines', []),
                'vertical_lines': turn_data.get('vertical_lines', []),
                'box_owners': turn_data.get('boxes', [])
            }
        
        if board_state:
            analysis['creates_third_edge'] = self.check_creates_third_edge(
                board_state, move_type, i, j
            )
            analysis['safe_move'] = not analysis['creates_third_edge']
        
        return analysis
    
    def check_creates_third_edge(self, board_state, move_type, i, j):
        """Check if a move creates a box with 3 edges (dangerous move)."""
        grid_size = board_state.get('grid_size', 4)
        h_lines = board_state.get('horizontal_lines', [])
        v_lines = board_state.get('vertical_lines', [])
        boxes = board_state.get('box_owners', [])
        
        # Simulate the move
        if move_type == 'h':
            h_lines[i][j] = 1  # temporarily mark
        else:
            v_lines[i][j] = 1
        
        creates_third = False
        
        # Check affected boxes
        affected_boxes = []
        if move_type == 'h':
            if i > 0:
                affected_boxes.append((i - 1, j))
            if i < grid_size:
                affected_boxes.append((i, j))
        else:  # vertical
            if j > 0:
                affected_boxes.append((i, j - 1))
            if j < grid_size:
                affected_boxes.append((i, j))
        
        for bi, bj in affected_boxes:
            if 0 <= bi < grid_size and 0 <= bj < grid_size:
                if boxes[bi][bj] is None:
                    edge_count = self.count_box_edges(h_lines, v_lines, bi, bj)
                    if edge_count == 3:
                        creates_third = True
                        break
        
        # Undo simulation
        if move_type == 'h':
            h_lines[i][j] = 0
        else:
            v_lines[i][j] = 0
        
        return creates_third
    
    def count_box_edges(self, h_lines, v_lines, bi, bj):
        """Count edges of a box."""
        count = 0
        if h_lines[bi][bj]:
            count += 1
        if h_lines[bi + 1][bj]:
            count += 1
        if v_lines[bi][bj]:
            count += 1
        if v_lines[bi][bj + 1]:
            count += 1
        return count
    
    def compute_quantitative_metrics(self):
        """Compute quantitative evaluation metrics."""
        print("\n" + "=" * 70)
        print("QUANTITATIVE EVALUATION")
        print("=" * 70)
        
        total_games = len(self.games)
        completed_games = [g for g in self.games if g['final_result']]
        
        print(f"\nGames Overview:")
        print(f"  Total games: {total_games}")
        print(f"  Completed games: {len(completed_games)}")
        
        # Win rate analysis
        vlm_wins = 0
        bot_wins = 0
        ties = 0
        
        vlm_total_score = 0
        bot_total_score = 0
        
        for game in completed_games:
            result = game['final_result']
            winner = result.get('winner')
            
            if winner == 1:  # VLM is player 1
                vlm_wins += 1
            elif winner == 0:  # Bot is player 0
                bot_wins += 1
            else:
                ties += 1
            
            final_scores = result.get('final_scores', {})
            vlm_total_score += final_scores.get('blue', 0)
            bot_total_score += final_scores.get('red', 0)
        
        print(f"\nWin Rate Analysis:")
        print(f"  VLM wins: {vlm_wins} ({vlm_wins/len(completed_games)*100:.1f}%)")
        print(f"  Bot wins: {bot_wins} ({bot_wins/len(completed_games)*100:.1f}%)")
        print(f"  Ties: {ties} ({ties/len(completed_games)*100:.1f}%)")
        
        print(f"\nScore Analysis:")
        print(f"  VLM total boxes captured: {vlm_total_score}")
        print(f"  Bot total boxes captured: {bot_total_score}")
        print(f"  VLM average boxes per game: {vlm_total_score/len(completed_games):.2f}")
        print(f"  Bot average boxes per game: {bot_total_score/len(completed_games):.2f}")
        
        # Move quality analysis
        vlm_moves_data = []
        bot_moves_data = []
        
        for game in self.games:
            for i, turn in enumerate(game['turns'][1:], 1):  # Skip first turn (initial state)
                prev_turn = game['turns'][i-1] if i > 0 else None
                
                # New format
                move = turn.get('move_taken')
                player = turn.get('move_player')
                
                # Old format - infer move and player from last_move
                if move is None:
                    move = turn.get('last_move')
                    # In old format, current_player is the player about to move
                    # So the player who made last_move is the other player
                    if move:
                        player = 1 - turn.get('current_player', 0)
                
                if move and player is not None:
                    analysis = self.analyze_move_quality(turn, move, prev_turn)
                    
                    if player == 1:  # VLM
                        vlm_moves_data.append(analysis)
                    else:  # Bot
                        bot_moves_data.append(analysis)
        
        print(f"\nMove Quality Analysis:")
        print(f"  Total VLM moves: {len(vlm_moves_data)}")
        print(f"  Total Bot moves: {len(bot_moves_data)}")
        
        if vlm_moves_data:
            vlm_box_completions = sum(1 for m in vlm_moves_data if m['completed_box'])
            vlm_third_edges = sum(1 for m in vlm_moves_data if m['creates_third_edge'])
            vlm_safe = sum(1 for m in vlm_moves_data if m['safe_move'])
            
            print(f"\n  VLM Statistics:")
            print(f"    Moves completing boxes: {vlm_box_completions} ({vlm_box_completions/len(vlm_moves_data)*100:.1f}%)")
            print(f"    Moves creating 3rd edge: {vlm_third_edges} ({vlm_third_edges/len(vlm_moves_data)*100:.1f}%)")
            print(f"    Safe moves: {vlm_safe} ({vlm_safe/len(vlm_moves_data)*100:.1f}%)")
        
        if bot_moves_data:
            bot_box_completions = sum(1 for m in bot_moves_data if m['completed_box'])
            bot_third_edges = sum(1 for m in bot_moves_data if m['creates_third_edge'])
            bot_safe = sum(1 for m in bot_moves_data if m['safe_move'])
            
            print(f"\n  Bot Statistics:")
            print(f"    Moves completing boxes: {bot_box_completions} ({bot_box_completions/len(bot_moves_data)*100:.1f}%)")
            print(f"    Moves creating 3rd edge: {bot_third_edges} ({bot_third_edges/len(bot_moves_data)*100:.1f}%)")
            print(f"    Safe moves: {bot_safe} ({bot_safe/len(bot_moves_data)*100:.1f}%)")
        
        # Phase-based analysis
        print(f"\n  Move Distribution by Game Phase:")
        for player_name, moves_data in [("VLM", vlm_moves_data), ("Bot", bot_moves_data)]:
            phase_counts = Counter(m['game_phase'] for m in moves_data)
            print(f"\n    {player_name}:")
            for phase in ['early', 'mid', 'late']:
                count = phase_counts.get(phase, 0)
                if len(moves_data) > 0:
                    print(f"      {phase.capitalize()}: {count} ({count/len(moves_data)*100:.1f}%)")
        
        return {
            'total_games': total_games,
            'completed_games': len(completed_games),
            'vlm_wins': vlm_wins,
            'bot_wins': bot_wins,
            'ties': ties,
            'vlm_moves': vlm_moves_data,
            'bot_moves': bot_moves_data
        }
    
    def perform_qualitative_analysis(self, metrics):
        """Perform qualitative analysis of VLM errors."""
        print("\n" + "=" * 70)
        print("QUALITATIVE ANALYSIS")
        print("=" * 70)
        
        vlm_moves = metrics['vlm_moves']
        
        # Identify problematic moves
        print("\n1. PROBLEMATIC MOVE PATTERNS")
        print("-" * 70)
        
        dangerous_moves = [m for m in vlm_moves if m['creates_third_edge']]
        print(f"\nDangerous Moves (creating 3rd edge):")
        print(f"  Count: {len(dangerous_moves)}")
        print(f"  Percentage: {len(dangerous_moves)/len(vlm_moves)*100:.1f}%")
        
        # Analyze when these occur
        phase_distribution = Counter(m['game_phase'] for m in dangerous_moves)
        print(f"  Distribution by phase:")
        for phase, count in phase_distribution.most_common():
            print(f"    {phase.capitalize()}: {count} ({count/len(dangerous_moves)*100:.1f}%)")
        
        # Analyze move patterns
        print(f"\n2. MOVE PATTERN ANALYSIS")
        print("-" * 70)
        
        move_types = Counter(m['move'][0] for m in vlm_moves)
        print(f"\nMove Type Distribution:")
        print(f"  Horizontal: {move_types.get('h', 0)} ({move_types.get('h', 0)/len(vlm_moves)*100:.1f}%)")
        print(f"  Vertical: {move_types.get('v', 0)} ({move_types.get('v', 0)/len(vlm_moves)*100:.1f}%)")
        
        # Position analysis
        positions = defaultdict(int)
        for m in vlm_moves:
            _, i, j = m['move']
            positions[f"({i},{j})"] += 1
        
        print(f"\nMost Frequent Positions (top 10):")
        for pos, count in sorted(positions.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {pos}: {count} times")
        
        # Sample problematic games
        print(f"\n3. SAMPLE PROBLEMATIC SCENARIOS")
        print("-" * 70)
        
        self.analyze_sample_errors(dangerous_moves[:5])
        
        # Performance by game phase
        print(f"\n4. PERFORMANCE BY GAME PHASE")
        print("-" * 70)
        
        for phase in ['early', 'mid', 'late']:
            phase_moves = [m for m in vlm_moves if m['game_phase'] == phase]
            if phase_moves:
                completed = sum(1 for m in phase_moves if m['completed_box'])
                dangerous = sum(1 for m in phase_moves if m['creates_third_edge'])
                
                print(f"\n{phase.capitalize()} Game:")
                print(f"  Total moves: {len(phase_moves)}")
                print(f"  Box completions: {completed} ({completed/len(phase_moves)*100:.1f}%)")
                print(f"  Dangerous moves: {dangerous} ({dangerous/len(phase_moves)*100:.1f}%)")
        
        # Common error types
        print(f"\n5. COMMON ERROR TYPES")
        print("-" * 70)
        
        error_analysis = self.categorize_errors(vlm_moves)
        for error_type, details in error_analysis.items():
            print(f"\n{error_type}:")
            print(f"  Count: {details['count']}")
            print(f"  Description: {details['description']}")
    
    def analyze_sample_errors(self, error_moves):
        """Analyze specific error examples."""
        if not error_moves:
            print("\nNo problematic moves to analyze.")
            return
        
        print("\nExample problematic moves:")
        for idx, move_data in enumerate(error_moves[:3], 1):
            print(f"\nExample {idx}:")
            print(f"  Move: {move_data['move']}")
            print(f"  Game Phase: {move_data['game_phase']}")
            print(f"  Created 3rd edge: {move_data['creates_third_edge']}")
            print(f"  Available moves: {move_data['available_moves_count']}")
    
    def categorize_errors(self, vlm_moves):
        """Categorize different types of errors."""
        errors = {}
        
        # Type 1: Creating third edges (giving opponent easy boxes)
        third_edge_moves = [m for m in vlm_moves if m['creates_third_edge']]
        errors['Third Edge Creation'] = {
            'count': len(third_edge_moves),
            'description': 'VLM creates boxes with 3 edges, giving opponent easy completion'
        }
        
        # Type 2: Missing box completion opportunities
        missed_completions = 0
        for game in self.games:
            for i, turn in enumerate(game['turns'][1:], 1):
                # Determine player (handle both old and new formats)
                player = turn.get('move_player')
                if player is None:
                    move = turn.get('last_move')
                    if move:
                        player = 1 - turn.get('current_player', 0)
                
                if player == 1:  # VLM turn
                    # Check if box was completed
                    completed = turn.get('strategy_info', {}).get('completed_boxes_this_turn')
                    if completed is None and i > 0:
                        # Calculate from score difference
                        prev_turn = game['turns'][i-1]
                        prev_scores = prev_turn.get('scores', {})
                        curr_scores = turn.get('scores', {})
                        prev_total = prev_scores.get('red', 0) + prev_scores.get('blue', 0)
                        curr_total = curr_scores.get('red', 0) + curr_scores.get('blue', 0)
                        completed = curr_total > prev_total
                    
                    if not completed:
                        # Check if there were 3-edge boxes available
                        potential = turn.get('strategy_info', {}).get('potential_boxes', {})
                        if potential.get('boxes_with_3_edges', 0) > 0:
                            missed_completions += 1
        
        errors['Missed Completions'] = {
            'count': missed_completions,
            'description': 'VLM failed to complete available boxes with 3 edges'
        }
        
        # Type 3: Early game aggression
        early_dangerous = len([m for m in third_edge_moves if m['game_phase'] == 'early'])
        errors['Early Game Mistakes'] = {
            'count': early_dangerous,
            'description': 'Dangerous moves made in early game when safety is crucial'
        }
        
        return errors
    
    def generate_summary_report(self, metrics):
        """Generate final summary report."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        vlm_win_rate = metrics['vlm_wins'] / metrics['completed_games'] * 100 if metrics['completed_games'] > 0 else 0
        
        print(f"\nOverall Performance:")
        print(f"  Games Analyzed: {metrics['total_games']}")
        print(f"  VLM Win Rate: {vlm_win_rate:.1f}%")
        print(f"  Total Moves Made: {len(metrics['vlm_moves'])}")
        
        if metrics['vlm_moves']:
            safe_rate = sum(1 for m in metrics['vlm_moves'] if m['safe_move']) / len(metrics['vlm_moves']) * 100
            completion_rate = sum(1 for m in metrics['vlm_moves'] if m['completed_box']) / len(metrics['vlm_moves']) * 100
            
            print(f"\nMove Quality:")
            print(f"  Safe Move Rate: {safe_rate:.1f}%")
            print(f"  Box Completion Rate: {completion_rate:.1f}%")
        
        print(f"\nKey Findings:")
        dangerous_count = sum(1 for m in metrics['vlm_moves'] if m['creates_third_edge'])
        print(f"  • VLM creates dangerous positions (3rd edges) in {dangerous_count} moves")
        print(f"  • This represents {dangerous_count/len(metrics['vlm_moves'])*100:.1f}% of all moves")
        
        if vlm_win_rate < 40:
            print(f"  • VLM performance is below baseline (random play ~50%)")
            print(f"  • Major improvement needed in strategic understanding")
        elif vlm_win_rate < 50:
            print(f"  • VLM performance is near random play level")
            print(f"  • Needs better move evaluation and planning")
        else:
            print(f"  • VLM shows strategic capability above random play")
            print(f"  • Further optimization could improve performance")
        
        print(f"\nRecommendations:")
        print(f"  1. Improve VLM's ability to recognize 3-edge boxes")
        print(f"  2. Train on more diverse game scenarios")
        print(f"  3. Implement better strategic prompting")
        print(f"  4. Add explicit safety checks in move selection")
        print(f"  5. Consider fine-tuning on Dots and Boxes specific data")


def main():
    """Main evaluation function."""
    print("=" * 70)
    print("VLM EVALUATION FOR DOTS AND BOXES")
    print("=" * 70)
    
    evaluator = DotsAndBoxesEvaluator()
    
    # Load all game data
    evaluator.load_all_games()
    
    # Compute quantitative metrics
    metrics = evaluator.compute_quantitative_metrics()
    
    # Perform qualitative analysis
    evaluator.perform_qualitative_analysis(metrics)
    
    # Generate summary
    evaluator.generate_summary_report(metrics)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
