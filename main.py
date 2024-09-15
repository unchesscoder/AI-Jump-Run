import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
from collections import deque

# Non-interactive backend for diagrams
matplotlib.use('Agg')

# Initialize Pygame
pygame.init()

# Screen parameters
WIDTH, HEIGHT = 800, 600
FPS = 60

# Colors
COLORS = {
    'WHITE': (255, 255, 255),
    'RED': (255, 0, 0),
    'GREEN': (0, 128, 0),
    'BLUE': (0, 0, 255),
}

# Player parameters
player_size = 50
player_speed = 8
player_y = HEIGHT - player_size * 2
player_x = WIDTH // 4

# Ground parameters
ground_height = 20

# Obstacles in the ground
min_obstacle_distance = 300
obstacle_frequency = 20
max_obstacle_size = 100
obstacles = []

# Score and game over status
score = 0
high_score = 0
game_over = False

# Replay buffer
replay_memory = deque(maxlen=10000)
batch_size = 64
gamma = 0.98  # Discount factor

# Reward parameters (constants for easy adjustment)
BASE_REWARD = 0.1       # Base reward for survival
OBSTACLE_CROSS_REWARD = 10.0  # Reward for crossing an obstacle
HIGHSCORE_REWARD = 50.0    # Reward for beating the high score
COLLISION_PENALTY = -50.0  # Penalty for collision
JUMP_PENALTY = -5.0        # Penalty for excessive jumping

# Neural network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 24)  # 6 input values
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 2)  # 2 actions: jump or don't jump

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Initialize model and optimizer
model = QNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Pygame screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jump and Run with AI")

# Font
font = pygame.font.Font(None, 36)

# Clock for FPS
clock = pygame.time.Clock()

# Function to normalize the state
def normalize_state(player_y, player_size, obstacles):
    if len(obstacles) == 0:
        obstacle1_x, obstacle1_width, obstacle2_x, obstacle2_width = 0, 0, 0, 0
    elif len(obstacles) == 1:
        obstacle1_x = obstacles[0].x / WIDTH
        obstacle1_width = obstacles[0].width / WIDTH
        obstacle2_x, obstacle2_width = 0, 0
    else:
        obstacle1_x = obstacles[0].x / WIDTH
        obstacle1_width = obstacles[0].width / WIDTH
        obstacle2_x = obstacles[1].x / WIDTH
        obstacle2_width = obstacles[1].width / WIDTH

    return torch.tensor([player_y / HEIGHT, player_size / WIDTH, obstacle1_x, obstacle1_width, obstacle2_x, obstacle2_width], dtype=torch.float32).view(1, -1)

# Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    else:
        with torch.no_grad():
            q_values = model(state)
            return torch.argmax(q_values).item()

# Restart game
def restart_game():
    global player_y, obstacles, score, game_over, is_jumping, steps
    player_y = HEIGHT - player_size * 2
    obstacles = []
    score = 0
    game_over = False
    is_jumping = False
    steps = 0  # Zurücksetzen der Schritte am Anfang jeder Episode

# Train the model
def train_model():
    if len(replay_memory) < batch_size:
        return 0  # Return value for loss

    mini_batch = random.sample(replay_memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*mini_batch)

    states = torch.cat(states)
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = model(states).gather(1, actions)
    next_q_values = model(next_states).max(1)[0].unsqueeze(1)
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = criterion(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Return the loss value

# Store game experience
def store_experience(state, action, reward, next_state, done):
    replay_memory.append((state, action, reward, next_state, done))

# Check for collision
def check_collision(player_rect, obstacles):
    for obstacle in obstacles:
        if player_rect.colliderect(obstacle):
            return True
    return False

# Check for the distance to the next obstacle
def distance_to_next_obstacle(player_x, obstacles):
    if len(obstacles) > 0:
        # Calculate the distance from the player to the first obstacle
        next_obstacle = obstacles[0]
        return next_obstacle.x - player_x
    return float('inf')  # If no obstacle, return a large value   

# Calculate reward including context-aware jump penalty
def calculate_reward(player_rect, prev_obstacles, current_obstacles, score, high_score, steps, action, distance_to_obstacle):
    reward = 0  # Start with 0 reward

    # Base reward for survival, scaled by steps
    reward += BASE_REWARD * steps

    # Reward for crossing an obstacle
    for prev_obstacle in prev_obstacles:
        if prev_obstacle not in current_obstacles:
            reward += OBSTACLE_CROSS_REWARD

    # High score reward
    if score > high_score:
        reward += HIGHSCORE_REWARD

    # Context-aware jump penalty: Only apply if the player jumps with no obstacle nearby
    if action == 1 and distance_to_obstacle > player_size * 2:  # Jump penalty if no nearby obstacle
        reward += JUMP_PENALTY

    return reward

# Update diagrams
def save_plots():
    plt.figure(figsize=(15, 10))  # Adjust figure size
    plt.subplot(2, 2, 1)
    plt.plot(total_rewards)
    plt.title("Total Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 2)
    plt.plot(avg_q_values)
    plt.title("Average Q-Value")
    plt.xlabel("Episodes")
    plt.ylabel("Average Q-Value")

    plt.subplot(2, 2, 3)
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 4)
    plt.plot(total_scores)
    plt.title("Score Progression")
    plt.xlabel("Episodes")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig('training_progress.png')
    print("Diagrams saved as 'training_progress.png'.")

# Main game loop
num_epochs = 2000000
is_jumping = False
jump_height = 100  # Fixed jump height
jump_velocity = -15  # Initial velocity for the jump
gravity = 1.0  # Gravity constant

# Variables for learning progress
total_rewards = []
avg_q_values = []
losses = []
total_scores = []  # Add score progression
epsilon = 1.0  # Starting value for epsilon
epsilon_decay = 0.9998  # Epsilon decay rate
min_epsilon = 0.01  # Minimum epsilon value
successful_episodes = 0  # Successful episodes
steps = 0  # Number of steps survived

for epoch in range(num_epochs):
    try:
        # Debug output
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}")

        total_loss = train_model()  # Calculate loss
        losses.append(total_loss)   # Store loss

        prev_obstacles = list(obstacles)  # Store previous obstacles

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not game_over:
            steps += 1  # Increase steps
            state = normalize_state(player_y, player_size, obstacles)

            # Calculate distance to the next obstacle
            distance_to_obstacle = distance_to_next_obstacle(player_x, obstacles)

            # Select action
            action = select_action(state, epsilon)

            # Average Q-value for the current state-action pair
            with torch.no_grad():
                q_values = model(state)
                avg_q_value = q_values.mean().item()

            avg_q_values.append(avg_q_value)

            if action == 1 and not is_jumping:
                is_jumping = True

        # Jumping logic
        if is_jumping:
            player_y += jump_velocity
            jump_velocity += gravity

            if player_y >= HEIGHT - ground_height - player_size:
                player_y = HEIGHT - ground_height - player_size
                is_jumping = False
                jump_velocity = -15
        else:
            player_y = HEIGHT - ground_height - player_size

        # Generate obstacles
        if random.randint(0, obstacle_frequency) == 0:
            if not obstacles or WIDTH - obstacles[-1].x >= min_obstacle_distance:
                obstacle_x = WIDTH

                # Obstacle width and height
                obstacle_width = random.randint(20, int(jump_height * 0.5))
                obstacle_height = random.randint(20, int(jump_height * 0.8))
                obstacle_y = HEIGHT - ground_height - obstacle_height
                obstacles.append(pygame.Rect(obstacle_x, obstacle_y, obstacle_width, obstacle_height))

        # Move obstacles
        for obstacle in obstacles:
            obstacle.x -= player_speed

        # Collision detection
        player_rect = pygame.Rect(player_x, player_y, player_size, player_size)
        next_state = normalize_state(player_y, player_size, obstacles)
        
        # Calculate distance to the next obstacle
        distance_to_obstacle = distance_to_next_obstacle(player_x, obstacles)
        
        # Calculate reward
        reward = calculate_reward(player_rect, prev_obstacles, obstacles, score, high_score, steps, action, distance_to_obstacle)

        if check_collision(player_rect, obstacles):
            game_over = True
            reward += COLLISION_PENALTY

        if is_jumping:
            reward += JUMP_PENALTY

        if not game_over:
            score += 1
            total_rewards.append(reward)
            total_scores.append(score)
            store_experience(state, action, reward, next_state, game_over)
        else:
            successful_episodes += 1
            if score > high_score:
                high_score = score
            reward += HIGHSCORE_REWARD
            store_experience(state, action, reward, next_state, game_over)

        obstacles = [obstacle for obstacle in obstacles if obstacle.x + obstacle.width > 0]

        loss = train_model()
        losses.append(loss)

        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, COLORS['WHITE'], [player_x, player_y, player_size, player_size])
        pygame.draw.rect(screen, COLORS['WHITE'], [0, HEIGHT - ground_height, WIDTH, ground_height])

        for obstacle in obstacles:
            pygame.draw.rect(screen, COLORS['RED'], obstacle)

        score_text = font.render(f"Score: {score}", True, COLORS['WHITE'])
        screen.blit(score_text, (10, 10))
        avg_q_text = font.render(f"Avg Q-Value: {avg_q_value:.2f}", True, COLORS['WHITE'])
        screen.blit(avg_q_text, (10, 50))

        if game_over:
            game_over_text = font.render(f"Game Over! Score: {score}", True, COLORS['WHITE'])
            screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 3))
            pygame.display.flip()
            pygame.time.wait(1)
            restart_game()

        pygame.display.flip()
        clock.tick(FPS)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if epoch % 1000 == 0:
            save_plots()

    except Exception as e:
        print(f"Error during epoch {epoch}: {e}")
        break

pygame.quit()
