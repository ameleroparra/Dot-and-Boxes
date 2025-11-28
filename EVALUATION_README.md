## Files Generated

### Reports
- **`EVALUATION_REPORT.md`** - Comprehensive analysis with all findings (MAIN REPORT)
- **`RESULTS_SUMMARY.py`** - Executive summary in text format
- **`game_results.csv`** - Game-by-game statistics (26 games)
- **`move_analysis.csv`** - Move-by-move data (1,040 moves)

### Evaluation Scripts
- **`evaluate_vlm.py`** - Quantitative evaluation (win rates, move quality)
- **`detailed_evaluation_report.py`** - Qualitative analysis (error patterns)
- **`export_results.py`** - Data export to CSV

## How to Run the Evaluation

```bash
# Run full quantitative evaluation
python evaluate_vlm.py

# Run detailed qualitative analysis
python detailed_evaluation_report.py

# Export data to CSV files
python export_results.py

# View summary
python RESULTS_SUMMARY.py
```
