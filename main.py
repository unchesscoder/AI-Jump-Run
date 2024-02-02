import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim

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

# Löcher im Boden
min_hole_distance = 200  # Mindestabstand zwischen Löchern
hole_frequency = 20
max_hole_size = 100
holes = []

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

# Liste für die Wolken
clouds = []

# Funktion zum Generieren einer Wolke
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

# Hauptspiel-Schleife
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
    if not is_jumping and not game_over:
        if keys[pygame.K_SPACE]:
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

    # Generiere neue Wolken
    if random.randint(0, 100) < 5:  # Geringe Wahrscheinlichkeit für jede Schleifeniteration
        clouds.append(generate_cloud())

    # Bewege die Wolken nach links
    for cloud in clouds:
        cloud.x -= player_speed // 2  # Wolken bewegen sich langsamer als der Spieler

    # Entferne Wolken, die den Bildschirm verlassen haben
    clouds = [cloud for cloud in clouds if cloud.x + cloud.width > 0]

    # Bildschirm leeren
    screen.fill(SKY_COLOR)

    # Zeichne Wolken
    draw_clouds()

    if not game_over:
        # Bewegung der Löcher
        for hole in holes:
            hole.x -= player_speed

        # Erzeugung neuer Löcher
        if random.randint(0, hole_frequency) == 0:
            generate_hole()
            score += 1  # Zähle den Punkt, wenn ein Loch generiert wird

        # Kollision mit Löchern
        for hole in holes:
            if hole.colliderect(pygame.Rect(player_x, player_y, player_size, player_size)):
                game_over = True

        # Löschen der alten Löcher
        holes = [hole for hole in holes if hole.x + hole.width > 0]

    # Zeichne Spieler, Boden, Löcher und den Punktezähler
    draw_player(player_x, player_y)
    draw_ground()
    draw_holes()
    draw_score()

    if game_over:
        restart_rect, exit_rect = draw_game_over()

    # Aktualisiere den Bildschirm
    pygame.display.flip()

    # Begrenze die Aktualisierungsrate
    clock.tick(FPS)