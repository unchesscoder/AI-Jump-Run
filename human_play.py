import pygame
import sys
import random

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
PLAYER_SIZE = 50
PLAYER_SPEED = 10
GROUND_HEIGHT = 0
OBSTACLE_MIN_DISTANCE = 200
OBSTACLE_FREQUENCY = 15

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jump and Run - Human Play")

# Load background image
background_image = pygame.image.load('./pictures/background.jpg')
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

# Load font
font = pygame.font.Font(None, 36)

# Reset game function
def reset_game():
    global player_y, player_velocity, obstacles, score, game_over, is_jumping
    player_y = HEIGHT - PLAYER_SIZE * 2
    player_velocity = 0
    obstacles.clear()
    score = 0
    game_over = False
    is_jumping = False

# Main game loop variables
is_jumping = False
jump_velocity = -15
gravity = 1.0
player_x = WIDTH / 2
player_y = HEIGHT - PLAYER_SIZE * 2
player_velocity = 0
obstacles = []
score = 0
high_score = 0
game_over = False

# Clock for FPS
clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not game_over:
        keys = pygame.key.get_pressed()

        # Player jump control
        if keys[pygame.K_SPACE] and not is_jumping:
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
        if any(player_rect.colliderect(obstacle) for obstacle in obstacles):
            game_over = True
            print(f"Game Over! Score: {score}")

        # Update display
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
        if score > high_score:
            high_score = score

    else:
        # Reset the game after the player loses
        reset_game()

pygame.quit()
