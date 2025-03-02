# Jump and Run with AI


![AI jump and run](https://github.com/user-attachments/assets/05935b6f-a2b0-4f04-bc1a-b708acffed2a)


## 1. Introduction

This project is a simple game built with Pygame and PyTorch, where an AI-controlled character navigates through obstacles by jumping. The objective is to maximize the score by avoiding collisions and surviving as long as possible.

## 2. Game Description

- **Player Character**:(GREEN MODEL) Can jump or stay grounded.
- **Obstacles**:(RED MODEL) Move from right to left. Their width is up to 50% of the jump height, and their height is up to 80% of the jump height.
- **Scoring**: Points are earned for surviving, crossing obstacles, and achieving high scores.

## 3. Technical Details

### Game Mechanics

- **Player Movement**: Controlled by jumping or not.
- **Obstacle Generation**: Obstacles appear based on a frequency and are sized relative to the jump height.
- **Collision Detection**: Checks if the player hits an obstacle.
- **Rewards and Penalties**: 
  - Base reward for survival.
  - Bonus for crossing obstacles.
  - Penalty for collisions and excessive jumping.
  - Reward for beating the high score.

### AI Model

- **Neural Network**: Uses a simple feedforward network to decide actions (jump or stay).
- **Training**: Utilizes experience replay and Q-learning to improve the model's performance.

## 4. Code Overview

The main components include:
- **Game Loop**: Handles player actions, obstacle movement, and collision detection.
- **Model Training**: Updates the neural network based on game experiences.
- **Visualization**: Saves plots of training progress, including rewards, Q-values, loss, and scores.

## 5. Dependencies

- **Pygame**: For game development.
- **PyTorch**: For the AI model.
- **Matplotlib**: For visualizing training progress.

## 6. Running the Game

1. Clone the repository.
2. Install the dependencies using `pip install pygame torch matplotlib`.
3. Run the main script to start the game and train the AI.

## 7. Future Work

- **Enhanced AI**: Experiment with more complex neural networks.
- **Additional Features**: Include different types of obstacles or power-ups.
- **Optimization**: Improve game performance and model efficiency.
