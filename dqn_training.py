import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import os

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
PLAYER_SIZE = 50
PLAYER_SPEED = 10  # Slightly faster player
GROUND_HEIGHT = 0
OBSTACLE_MIN_DISTANCE = 500  # Slightly closer obstacles
OBSTACLE_FREQUENCY = 15  # More frequent obstacles
REPLAY_BUFFER_SIZE = 15000  # Increased buffer size for diversity
BATCH_SIZE = 32  # Smaller batch size for frequent updates
GAMMA = 0.99
BASE_REWARD = 1.0  # Small positive reward for staying alive
OBSTACLE_CROSS_REWARD = 50.0  # Increased reward for crossing obstacles
HIGHSCORE_REWARD = 5.0  # Increased reward for new high score
COLLISION_PENALTY = -5.0  # Reduced penalty for collisions
JUMP_PENALTY = -0.005  # Further reduced penalty for jumping
MIN_EPSILON = 0.05  # Further exploration allowed
EPSILON_DECAY = 0.97  # Slower decay for longer exploration
TARGET_UPDATE = 300  # Less frequent updates to stabilize learning

# Neural network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(7, 512)  # Increased hidden layer size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)  # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# Initialize model, target model, and optimizer
model = QNetwork()
target_model = QNetwork()
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate
criterion = nn.SmoothL1Loss()  # Huber loss for more stable training
model_directory = 'trained_models'

# Pfad für das Modell im Ordner 'trained_models'
model_directory = './trained_models/'
model_filename = os.path.join(model_directory, 'dqn_model_zero.pth')

# Erstelle den Ordner, falls er nicht existiert
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Überprüfe, ob das Modell existiert, und lade es oder erstelle ein neues
if os.path.exists(model_filename):
    model.load_state_dict(torch.load(model_filename))
    print(f"Loaded existing model from {model_filename}.")
else:
    # Falls das Modell nicht existiert, speichere ein neues Initialmodell
    torch.save(model.state_dict(), model_filename)
    print(f"No existing model found. Created new model as {model_filename}.")

# Pygame screen
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jump and Run with AI")

# Clock for FPS
clock = pygame.time.Clock()

# Load background image
background_image = pygame.image.load('./pictures/background.jpg')
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

# Load font
font = pygame.font.Font(None, 36)

# Function to normalize the state
def normalize_state(player_y, player_velocity, player_x, obstacles):
    obstacle1_x, obstacle1_width, obstacle2_x, obstacle2_width = 0, 0, 0, 0
    if len(obstacles) > 0:
        obstacle1_x = obstacles[0].x / WIDTH
        obstacle1_width = obstacles[0].width / WIDTH
    if len(obstacles) > 1:
        obstacle2_x = obstacles[1].x / WIDTH
        obstacle2_width = obstacles[1].width / WIDTH
    return torch.tensor([player_y / HEIGHT, player_velocity, player_x / WIDTH, obstacle1_x, obstacle1_width, obstacle2_x, obstacle2_width], dtype=torch.float32).view(1, -1)

# Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    else:
        with torch.no_grad():
            q_values = model(state)
            return torch.argmax(q_values).item()

# Reset game
def reset_game():
    global player_y, player_velocity, obstacles, score, game_over, is_jumping, steps, episode_score
    player_y = HEIGHT - PLAYER_SIZE * 2
    player_velocity = 0
    obstacles.clear()
    score = 0
    game_over = False
    is_jumping = False
    steps = 0
    episode_score = 0

# Train the model
def train_model():
    if len(replay_memory) < BATCH_SIZE:
        return 0

    mini_batch = random.sample(replay_memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*mini_batch)

    states = torch.cat(states)
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    # Compute Q-values
    q_values = model(states).gather(1, actions)  # Q-values for chosen actions
    next_q_values = target_model(next_states).max(1)[0].unsqueeze(1)  # Max Q-value of next state
    target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))  # Compute the target Q-value

    # Calculate loss and update model
    loss = criterion(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Store game experience
def store_experience(state, action, reward, next_state, done):
    replay_memory.append((state, action, reward, next_state, torch.tensor(done, dtype=torch.float32)))

# Check for collision
def check_collision(player_rect, obstacles):
    for obstacle in obstacles:
        if player_rect.colliderect(obstacle):
            return True
    return False

def calculate_reward(player_rect, prev_obstacles, current_obstacles, score, high_score, action):
    reward = 0
    if action == 1:  # If jumping
        reward += JUMP_PENALTY
    
    if len(prev_obstacles) < len(current_obstacles):
        if score > high_score:  # If score is greater than high score
            reward += OBSTACLE_CROSS_REWARD * 2  # Double the reward for crossing an obstacle
        else:
            reward += OBSTACLE_CROSS_REWARD  # Normal reward for crossing an obstacle

    if score > high_score:  # New high score
        reward += HIGHSCORE_REWARD * 2  # Double reward for exceeding high score
    elif score == high_score and not game_over:  # If it matches the high score
        reward += HIGHSCORE_REWARD  # Reward for matching the high score

    if check_collision(player_rect, obstacles):
        reward += COLLISION_PENALTY
        reward -= score  # Deduct score for collision

    reward += 0.1  # Survival reward
    reward += 0.01  # Small reward for each frame survived

    return reward

# Main game loop
num_epochs = 900000
is_jumping = False
jump_height = 100
jump_velocity = -15
gravity = 1.0

# Variables for learning progress
total_rewards = []
avg_q_values = []
losses = []
episode_rewards = []
episode_scores = []
epsilon = 1.0  # Start with high exploration
replay_memory = deque(maxlen=REPLAY_BUFFER_SIZE)
player_x = WIDTH / 2
player_y = HEIGHT - PLAYER_SIZE * 2
player_velocity = 0
obstacles = []
score = 0
high_score = 0
game_over = False
steps = 0
episode_score = 0

# Initialize lists for plotting
loss_history = []
score_history = []
high_score_history = []
avg_q_value_history = []
reward_history = []

# Main game loop
for epoch in range(num_epochs):
    try:
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}")

        prev_obstacles = list(obstacles)
        avg_q_value = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not game_over:
            steps += 1
            state = normalize_state(player_y, player_velocity, player_x, obstacles)
            action = select_action(state, epsilon)

            with torch.no_grad():
                q_values = model(state)
                avg_q_value = q_values.mean().item()

            if action == 1 and not is_jumping:
                is_jumping = True

            if is_jumping:
                player_velocity += gravity
                player_y += player_velocity

                if player_y >= HEIGHT - GROUND_HEIGHT - PLAYER_SIZE:
                    player_y = HEIGHT - GROUND_HEIGHT - PLAYER_SIZE
                    is_jumping = False
                    player_velocity = -15
            else:
                player_y = HEIGHT - GROUND_HEIGHT - PLAYER_SIZE

            # Generate obstacles
            if random.randint(0, OBSTACLE_FREQUENCY) == 0:
                if not obstacles or WIDTH - obstacles[-1].x >= OBSTACLE_MIN_DISTANCE:
                    obstacle_width = random.randint(30, 70)
                    obstacle = pygame.Rect(WIDTH, HEIGHT - GROUND_HEIGHT - PLAYER_SIZE, obstacle_width, PLAYER_SIZE)
                    obstacles.append(obstacle)

            # Update obstacle positions
            for obstacle in obstacles:
                obstacle.x -= PLAYER_SPEED

            if obstacles and obstacles[0].x < 0:
                obstacles.pop(0)  # Remove obstacles that are out of bounds

            # Collision detection
            player_rect = pygame.Rect(player_x, player_y, PLAYER_SIZE, PLAYER_SIZE)
            reward = calculate_reward(player_rect, prev_obstacles, obstacles, score, high_score, action)

            if check_collision(player_rect, obstacles):
                game_over = True
                print("Game Over!")

            # Store the experience
            next_state = normalize_state(player_y, player_velocity, player_x, obstacles)
            done = 1 if game_over else 0
            store_experience(state, action, reward, next_state, done)

            # Train the model
            loss = train_model()
            if loss is not None:
                losses.append(loss)

            # Update the high score
            if score > high_score:
                high_score = score

            # Update the display
            screen.blit(background_image, (0, 0))
            pygame.draw.rect(screen, (0, 255, 0), player_rect)
            for obstacle in obstacles:
                pygame.draw.rect(screen, (255, 0, 0), obstacle)

            score_text = font.render(f'Score: {score}', True, (255, 255, 255))
            high_score_text = font.render(f'High Score: {high_score}', True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            screen.blit(high_score_text, (10, 50))

            pygame.display.flip()
            clock.tick(FPS)

            # Increment score
            score += 1
            episode_score += 1  # Track episode score

            # Decay epsilon
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

            # Print metrics for tracking
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Score: {score}, High Score: {high_score}, Loss: {loss:.4f}, Epsilon: {epsilon:.4f}, Avg Q: {avg_q_value:.4f}, Reward: {reward:.4f}")

            # Save model every 500 epochs
            if epoch % 20000 == 0:
                model_filename = os.path.join(model_directory,f'dqn_model_epoch_{epoch}.pth')
                torch.save(model.state_dict(), model_filename)
                print(f"Model saved at epoch {epoch}")

            # Speichern des finalen Modells nach der letzten Epoche
            if epoch == num_epochs - 1:  # Letzte Epoche
                final_model_filename = os.path.join(model_directory, 'dqn_model_final.pth')
                torch.save(model.state_dict(), final_model_filename)
                print(f"Final model saved as {final_model_filename}")

            # Update target network
            if epoch % TARGET_UPDATE == 0:
                target_model.load_state_dict(model.state_dict())

        else:
            # Store episode metrics
            score_history.append(episode_score)
            high_score_history.append(high_score)
            loss_history.append(loss)
            avg_q_value_history.append(avg_q_value)
            reward_history.append(episode_score)  # Using episode score as reward

            # Reset game if game over
            reset_game()

    except Exception as e:
        print(f"Error in epoch {epoch}: {e}")

# Save the final model
torch.save(model.state_dict(), model_filename)
print("Final model saved.")

# After the game ends, plot the diagrams
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(loss_history)
plt.title('Loss over time')
plt.xlabel('Episodes')
plt.ylabel('Loss')

plt.subplot(3, 2, 2)
plt.plot(score_history)
plt.title('Score over time')
plt.xlabel('Episodes')
plt.ylabel('Score')

plt.subplot(3, 2, 3)
plt.plot(high_score_history)
plt.title('High Score over time')
plt.xlabel('Episodes')
plt.ylabel('High Score')

plt.subplot(3, 2, 4)
plt.plot(avg_q_value_history)
plt.title('Average Q-Value over time')
plt.xlabel('Episodes')
plt.ylabel('Average Q-Value')

plt.subplot(3, 2, 5)
plt.plot(reward_history)  # This will now display correctly
plt.title('Episode Reward over time')
plt.xlabel('Episodes')
plt.ylabel('Reward')

plt.tight_layout()
plt.show()

pygame.quit()
