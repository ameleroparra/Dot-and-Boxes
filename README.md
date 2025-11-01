# Dots and Boxes Game

A Dots and Boxes game implementation with screenshot capture functionality for dataset collection.

## Installation

```bash
pip install pygame
```

## Usage

Run the game:
```bash
python main.py
```

## Screenshot Feature

The game automatically captures screenshots after each move:

### Path Pattern
Screenshots are saved in the project folder:
```
dots-and-boxes/data/raw/game_{game_id}/turn_{turn}.png
```

### Example Structure
```
dots-and-boxes/
├── main.py
├── README.md
└── data/
    └── raw/
        ├── game_1/
        │   ├── turn_001.png  # Initial board state
        │   ├── turn_002.png  # After first move
        │   ├── turn_003.png  # After second move
        │   └── ...
        ├── game_2/
        │   └── ...
        └── game_3/
            └── ...
```

### Implementation
- Screenshots are saved after rendering each move
- Uses `pygame.image.save(screen, path)` 
- Directory is created automatically inside the project folder
- Game ID auto-increments (game_1, game_2, game_3...)
- Turn counter increments with each screenshot

## Game Features

- **1 vs 1 mode**: Two human players
- **1 vs AI mode**: Coming soon
- **Automatic screenshot capture**: Every move is saved with chronological numbering
- **Auto-incrementing game folders**: Each new game gets its own folder