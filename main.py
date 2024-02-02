import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Define your DQN model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=6, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=2)  # 2 actions: jump or do nothing

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialization
pygame.init()

# Screen parameters
WIDTH, HEIGHT = 800, 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
SKY_COLOR = (135, 206, 250)  # Light Blue

# Player parameters
player_size = 50
player_x, player_y = WIDTH // 4, HEIGHT - player_size * 2
player_speed = 8
is_jumping, jump_count = False, 10

# Ground parameters
ground_height = 20

# Holes in the ground
min_hole_distance, hole_frequency, max_hole_size = 200, 20, 100
holes = []

# Score counter
score = 0

# Game Over status
game_over = False

# Screen initialization
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jump and Run Game")

# Font for the score and game over message
font = pygame.font.Font(None, 36)

# Clock object for screen updates
clock = pygame.time.Clock()

# DQN parameters
state_size = 6  # Adjust as needed based on your state representation
action_size = 2  # Jump or do nothing
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration-exploitation trade-off
epsilon_decay = 0.995
epsilon_min = 0.01
memory = []
batch_size = 32  # Added batch size

# Initialize DQN model and optimizer
dqn_model = DQN()
optimizer = optim.Adam(dqn_model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

# Convert game state to a tensor
def get_state():
    # Player state
    player_state = torch.tensor([player_x, player_y], dtype=torch.float32)

    # Holes state
    holes_state = torch.tensor([[hole.x, hole.y] for hole in holes], dtype=torch.float32)
    
    
    # Pad smaller tensors to match the size of the largest one
    max_size = max(len(player_state), holes_state.size(0))
    if len(player_state) < max_size:
        player_state = torch.cat([player_state, torch.zeros(max_size - len(player_state))])
    if holes_state.size(0) < max_size:
        zeros_to_add = max_size - holes_state.size(0)
        holes_state = torch.cat([holes_state, torch.zeros(zeros_to_add, 2)])

    # Concatenate player and holes states into the final state tensor
    state = torch.cat([player_state, holes_state.view(-1)])

    return state




# Implement the DQN agent
def dqn_agent(state):
    global epsilon

    # Epsilon-greedy policy
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_size - 1)  # Explore
    else:
        with torch.no_grad():
            q_values = dqn_model(state)
            return torch.argmax(q_values).item()  # Exploit

def draw_player(x, y):
    # Kopf
    pygame.draw.circle(screen, WHITE, (x + player_size // 2, y), player_size // 4)

    # Körper
    pygame.draw.line(screen, WHITE, (x + player_size // 2, y + player_size // 4), (x + player_size // 2, y + player_size // 2 + player_size // 4), 2)

    # Arme
    pygame.draw.line(screen, WHITE, (x, y + player_size // 4 + player_size // 8), (x + player_size, y + player_size // 4 + player_size // 8), 2)

    # Beine
    pygame.draw.line(screen, WHITE, (x + player_size // 2, y + player_size // 2 + player_size // 4), (x, y + player_size), 2)
    pygame.draw.line(screen, WHITE, (x + player_size // 2, y + player_size // 2 + player_size // 4), (x + player_size, y + player_size), 2)

def draw_ground():
    pygame.draw.rect(screen, WHITE, [0, HEIGHT - ground_height, WIDTH, ground_height])

def draw_holes():
    for hole in holes:
        pygame.draw.rect(screen, (0, 0, 0), hole)

def draw_score():
    score_text = font.render("Punkte: {}".format(score), True, WHITE)
    screen.blit(score_text, (10, 10))

def draw_game_over():
    game_over_text = font.render("Game Over! Punkte: {}".format(score), True, WHITE)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 3))

    restart_text = font.render("Restart", True, WHITE)
    exit_text = font.render("Exit", True, WHITE)

    restart_rect = pygame.Rect(WIDTH // 3, 2 * HEIGHT // 3, restart_text.get_width(), restart_text.get_height())
    exit_rect = pygame.Rect(2 * WIDTH // 3 - exit_text.get_width(), 2 * HEIGHT // 3, exit_text.get_width(), exit_text.get_height())

    pygame.draw.rect(screen, (0, 128, 0), restart_rect)
    pygame.draw.rect(screen, (128, 0, 0), exit_rect)

    screen.blit(restart_text, (WIDTH // 3, 2 * HEIGHT // 3))
    screen.blit(exit_text, (2 * WIDTH // 3 - exit_text.get_width(), 2 * HEIGHT // 3))

    return restart_rect, exit_rect

def generate_hole():
    if not holes or WIDTH - holes[-1].x >= min_hole_distance:
        hole_x = WIDTH
        hole_width = random.randint(20, max_hole_size)  # Zufällige Breite, begrenzt durch max_hole_size
        hole_height = random.randint(20, max_hole_size)  # Zufällige Höhe, begrenzt durch max_hole_size
        hole_y = HEIGHT - ground_height - hole_height  # Platzierung am Boden
        holes.append(pygame.Rect(hole_x, hole_y, hole_width, hole_height))

# Farbe für den Himmel
SKY_COLOR = (135, 206, 250)  # Lichtblau

clouds= []

def generate_cloud():
    cloud_x = WIDTH
    cloud_y = random.randint(50, 200)  # Y-Position der Wolke
    cloud_size = random.randint(20, 50)  # Größe der Wolke
    return pygame.Rect(cloud_x, cloud_y, cloud_size, cloud_size)

# Funktion zum Zeichnen der Wolken
def draw_clouds():
    for cloud in clouds:
        pygame.draw.ellipse(screen, WHITE, cloud)

# Funktion zum Überprüfen, ob die Maus auf einen Button zeigt
def is_mouse_on_button(mouse_pos, button_rect):
    return button_rect.collidepoint(mouse_pos)

# Initialize restart_rect and exit_rect outside of the game loop
restart_rect, exit_rect = None, None

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and game_over:
            mouse_pos = pygame.mouse.get_pos()
            if is_mouse_on_button(mouse_pos, restart_rect):
                # Restart
                player_y = HEIGHT - player_size * 2
                holes = []
                score = 0
                game_over = False
            elif is_mouse_on_button(mouse_pos, exit_rect):
                pygame.quit()
                sys.exit()

    keys = pygame.key.get_pressed()
    if not is_jumping and not game_over and keys[pygame.K_SPACE]:
        is_jumping = True

    if is_jumping:
        if jump_count >= -10:
            neg = 1 if jump_count >= 0 else -1
            player_y -= (jump_count ** 2) * 0.5 * neg
            jump_count -= 1
        else:
            is_jumping, jump_count = False, 10

    # Generate new clouds
    if random.randint(0, 100) < 5:
        clouds.append(generate_cloud())

    # Move clouds to the left
    clouds = [pygame.Rect(cloud.x - player_speed // 2, cloud.y, cloud.width, cloud.height) for cloud in clouds if cloud.x + cloud.width > 0]

    # Clear the screen
    screen.fill(SKY_COLOR)

    # Draw clouds
    draw_clouds()

    state = get_state()

    if not game_over:
        # DQN agent makes a decision
        action = dqn_agent(state)

        # Execute the action
        if action == 0 and not is_jumping:
            is_jumping = True

        # Move holes
        holes = [hole.move(-player_speed, 0) for hole in holes]

        # Generate new holes
        if random.randint(0, hole_frequency) == 0:
            generate_hole()
            score += 1

        # Collision with holes
        game_over = any(hole.colliderect(pygame.Rect(player_x, player_y, player_size, player_size)) for hole in holes)

        # Remove old holes
        holes = [hole for hole in holes if hole.x + hole.width > 0]

        # Get the next state
        next_state = get_state()

        # Define reward
        reward = 1 if not game_over else -1

        # Store the experience in the replay memory
        memory.append((state, action, reward, next_state, game_over))

        # Update the DQN model
        if len(memory) >= batch_size:
           # Sample a random batch from the replay memory
            batch = random.sample(memory, batch_size)

            # Unpack the batch
            states, actions, rewards, next_states, dones = zip(*batch)

                        # Convert to PyTorch tensors
            states = torch.stack([torch.flatten(state) for state in states])
            next_states = torch.stack([torch.flatten(state) for state in next_states])

            # Find the maximum size
            max_size = max(states.size(1), next_states.size(1))

            # Pad the tensors with zeros to match the size of the largest one
            states = torch.cat([states, torch.zeros(states.size(0), max_size - states.size(1))], dim=1)
            next_states = torch.cat([next_states, torch.zeros(next_states.size(0), max_size - next_states.size(1))], dim=1)


            dones = torch.tensor(dones, dtype=torch.float32)
            
            # Compute Q-values for the current and next states
            q_values = dqn_model(states)
            next_q_values = dqn_model(next_states)
            print("next_q_values shape:", next_q_values.shape)
            print("dones shape:", dones.shape)
            print("torch.max(...).values shape:", torch.max(next_q_values, dim=1).values.shape)

            # Compute the target Q-values
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            target_q_values = rewards + gamma * (1 - dones) * torch.max(next_q_values, dim=1)[0]  # Removed unsqueeze(1)

            # Get the Q-values for the selected actions
            actions = torch.tensor(actions, dtype=torch.long)
            selected_q_values = torch.gather(q_values, 1, actions.unsqueeze(1))

            # Compute the loss
            loss = loss_function(selected_q_values, target_q_values.unsqueeze(1))

            # Update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
    
    # Draw player, ground, holes, and score
    draw_player(player_x, player_y)
    draw_ground()
    draw_holes()
    draw_score()

    if game_over:
        restart_rect, exit_rect = draw_game_over()

    # Update the screen
    pygame.display.flip()

    # Limit the update rate
    clock.tick(FPS)