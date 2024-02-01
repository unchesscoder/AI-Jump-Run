import pygame
import sys
import random
import numpy as np

# Initialisierung von Pygame
pygame.init()

# Bildschirmparameter
WIDTH, HEIGHT = 800, 600
FPS = 60

# Farben
WHITE = (255, 255, 255)

# Spielerparameter
player_size = 50
player_x = WIDTH // 4
player_y = HEIGHT - player_size * 2
player_speed = 8
is_jumping = False
jump_count = 10

# Bodenparameter
ground_height = 20

# Punktezähler
score = 0

# Game Over-Status
game_over = False

# Initialisierung des Bildschirms
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jump and Run Spiel")

# Font für den Punktezähler und Game Over-Nachricht
font = pygame.font.Font(None, 36)

# Clock-Objekt für die Aktualisierung des Bildschirms
clock = pygame.time.Clock()

# Funktion zum Zeichnen des Strichmännchens
def draw_player(x, y):
    pygame.draw.circle(screen, WHITE, (x + player_size // 2, y), player_size // 4)
    pygame.draw.line(screen, WHITE, (x + player_size // 2, y + player_size // 4), (x + player_size // 2, y + player_size // 2 + player_size // 4), 2)
    pygame.draw.line(screen, WHITE, (x, y + player_size // 4 + player_size // 8), (x + player_size, y + player_size // 4 + player_size // 8), 2)
    pygame.draw.line(screen, WHITE, (x + player_size // 2, y + player_size // 2 + player_size // 4), (x, y + player_size), 2)
    pygame.draw.line(screen, WHITE, (x + player_size // 2, y + player_size // 2 + player_size // 4), (x + player_size, y + player_size), 2)

def draw_ground():
    pygame.draw.rect(screen, WHITE, [0, HEIGHT - ground_height, WIDTH, ground_height])

def draw_score():
    score_text = font.render("Punkte: {}".format(score), True, WHITE)
    screen.blit(score_text, (10, 10))

def draw_game_over():
    game_over_text = font.render("Game Over! Punkte: {}".format(score), True, WHITE)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 3))

# Keras-Modell für die KI
model = Sequential()
model.add(Dense(24, input_dim=3, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer='adam')

def get_state():
    # Hier könnten Zustandsinformationen wie die Position des Spielers, Hindernisse, etc. extrahiert werden
    return [player_x, player_y, is_jumping]

# Hauptspiel-Schleife
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Input für das Keras-Modell
    state = np.array([get_state()])
    
    # Aktionen der KI
    q_values = model.predict(state)
    action = np.argmax(q_values[0])
    
    # Aktionen ausführen
    if action == 1 and not is_jumping:
        is_jumping = True

    if is_jumping:
        if jump_count >= -10:
            neg = 1
            if jump_count < 0:
                neg = -1
            player_y -= (jump_count ** 2) * 0.5 * neg
            jump_count -= 1
        else:
            is_jumping = False
            jump_count = 10

    # Bewegung des Spielers
    player_x += player_speed

    # Punktezähler aktualisieren
    score += 1

    # Überprüfen auf Game Over
    if player_y >= HEIGHT - ground_height - player_size:
        game_over = True

    # Bildschirm leeren
    screen.fill(WHITE)

    # Zeichne Spieler, Boden und den Punktezähler
    draw_player(player_x, player_y)
    draw_ground()
    draw_score()

    if game_over:
        draw_game_over()

    # Aktualisiere den Bildschirm
    pygame.display.flip()

    # Begrenze die Aktualisierungsrate
    clock.tick(FPS)
