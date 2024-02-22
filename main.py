import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Verwende Non-Interactive Backend für die Verwendung in nicht-GUI-Umgebungen
from threading import Thread

# Initialisierung von Pygame
pygame.init()

# Bildschirmparameter
WIDTH, HEIGHT = 800, 600
FPS = 60

# Farben
COLORS = {
    'WHITE': (255, 255, 255),
    'RED': (255, 0, 0),
    'GREEN': (0, 128, 0),
    'BLUE': (0, 0, 255),
}

# Spielerparameter
player_size = 50
player_speed = 8
is_jumping = False
jump_count = 10
player_y = HEIGHT - player_size * 2  # Definiere player_y hier
player_x = WIDTH // 4  # Definiere player_x hier

# Bodenparameter
ground_height = 20

# Löcher im Boden
min_hole_distance = 300  # Mindestabstand zwischen Löchern
hole_frequency = 20
max_hole_size = 100
holes = []

# Punktezähler
score = 0
high_score = 0

# Game Over-Status
game_over = False

# Trainingsdaten für die KI
training_data = []

# Neuronales Netzwerk für die Q-Learning-Steuerung
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # Eingabe: Spielerposition, Spielergröße, Lochpositionen
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 2)  # Ausgabe: Springen oder nicht springen

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialisierung des neuronalen Netzwerks, Optimierers und Verlustfunktion
model = QNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Erhöhung der Lernrate auf 0.01
criterion = nn.MSELoss()

# Initialisierung des Bildschirms
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jump and Run Spiel")

# Font für den Punktezähler und Game Over-Nachricht
font = pygame.font.Font(None, 36)

# Clock-Objekt für die Aktualisierung des Bildschirms
clock = pygame.time.Clock()

# Funktion zur Normalisierung von Zustandsinformationen für das neuronale Netzwerk
def normalize_state(player_y, player_size, holes):
    normalized_state = [
        player_y / HEIGHT,
        player_size / player_size,
        holes[0].x / WIDTH if holes else 0,
        holes[0].y / HEIGHT if holes else 0,
        holes[0].width / WIDTH if holes else 0,
    ]
    return torch.tensor(normalized_state, dtype=torch.float32).view(1, -1)

# Funktion zum Wählen einer Aktion basierend auf dem epsilon-greedy-Ansatz
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)  # Zufällige Aktion wählen
    else:
        with torch.no_grad():
            q_values = model(state)
            return torch.argmax(q_values).item()  # Aktion mit höchstem Q-Wert wählen

# Funktion zum Neustarten des Spiels
def restart_game():
    global player_y, holes, score, game_over
    player_y = HEIGHT - player_size * 2
    holes = []
    score = 0
    game_over = False
    is_jumping = False  # Setze is_jumping zurück

# Funktion zum Trainieren des Modells
def train_model():
    global model, optimizer, criterion, holes
    if len(training_data) > 0:
        # Trainingsdaten in Tensor konvertieren
        states, actions, rewards = zip(*training_data)
        states = torch.cat(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Modell trainieren
        optimizer.zero_grad()
        q_values = model(states)
        predicted_q_values = torch.gather(q_values, 1, actions.unsqueeze(1))
        target_q_values = rewards.unsqueeze(1)
        loss = criterion(predicted_q_values, target_q_values)
        loss.backward()
        optimizer.step()



# Funktion für das Diagramm-Update
def update_plot(score_plot, epoch):
    if epoch % 100 == 0:  # Überprüfe, ob die aktuelle Epoche durch 100 teilbar ist
        plt.plot(score_plot)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Training Score Progress')
        plt.grid(True)
        plt.savefig('score_plot.png')  # Speichern Sie das Diagramm als Bilddatei
        plt.close()

# Thread für das Diagramm-Update
score_plot = []
plot_thread = Thread(target=update_plot, args=(score_plot,))
plot_thread.daemon = True
plot_thread.start()

# Funktion zum Speichern des Modells
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

# Funktion zum Laden des Modells
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))

# Dateipfad zum Speichern und Laden des Modells
model_filepath = "q_learning_model.pth"

# Anzahl der Trainingsepochen
num_epochs = 200000

# Hauptspiel-Schleife
for epoch in range(num_epochs):
    print("Epoch:", epoch)

    total_loss = 0  # Initialisieren Sie den Gesamtverlust für diese Epoche
    total_points = 0  # Initialisieren Sie die gesammelten Punkte für diese Epoche
    current_reward = 0  # Initialisieren Sie die aktuelle Belohnung für diese Epoche

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if not game_over:
        # Aktion wählen
        epsilon = max(0.01, 0.08 - score * 0.001)  # Epsilon verringern im Laufe der Zeit
        state = normalize_state(player_y, player_size, holes)
        action = select_action(state, epsilon)

        # Aktion ausführen und Spielzustand aktualisieren
        if action == 1 and not is_jumping:
            is_jumping = True

    # Spiellogik aktualisieren
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
    else:
        # Spieler auf dem Boden halten
        player_y = HEIGHT - ground_height - player_size

    # Generiere neue Löcher
    if random.randint(0, hole_frequency) == 0:
        if not holes or WIDTH - holes[-1].x >= min_hole_distance:
            hole_x = WIDTH
            hole_width = random.randint(20, max_hole_size)
            hole_height = random.randint(20, max_hole_size)
            hole_y = HEIGHT - ground_height - hole_height
            holes.append(pygame.Rect(hole_x, hole_y, hole_width, hole_height))
            if not game_over:  # Zähle die Punktzahl, wenn das Spiel noch nicht vorbei ist
                score += 1
                if score > high_score:
                    high_score = score
                    current_reward = 1  # Belohnung für Verbesserung des Highscores
                else:
                    current_reward = 0  # Keine Belohnung, wenn der Highscore nicht verbessert wird

    # Bewege die Löcher nach links
    for hole in holes:
        hole.x -= player_speed

    # Kollision mit Löchern
    for hole in holes:
        if hole.colliderect(pygame.Rect(player_x, player_y, player_size, player_size)):
            game_over = True
            if not game_over:
                current_reward = -1  # Strafe für Kollision
                training_data.append((state, action, current_reward))  # Füge Trainingsdaten hinzu, wenn Spiel noch läuft
            break  # Die Schleife beenden, wenn eine Kollision erkannt wurde

    # Entferne Löcher, die den Bildschirm verlassen haben
    holes = [hole for hole in holes if hole.x + hole.width > 0]

    # Modell trainieren
    train_model()

    # Berechnen Sie den Verlust und fügen Sie ihn dem Gesamtverlust hinzu
    if len(training_data) > 0:
        states, actions, rewards = zip(*training_data)
        states = torch.cat(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        q_values = model(states)
        predicted_q_values = torch.gather(q_values, 1, actions.unsqueeze(1))
        target_q_values = rewards.unsqueeze(1)
        loss = criterion(predicted_q_values, target_q_values)
        total_loss += loss.item()  # Fügen Sie den aktuellen Verlust hinzu

    # Ausgabe von Gesamtverlust und durchschnittlicher Belohnung
    average_reward = total_points / (epoch + 1)  # Berechnen Sie die durchschnittliche Belohnung pro Epoche
    print("Total Loss (Epoch {}): {:.4f}".format(epoch, total_loss))
    print("Average Reward (Epoch {}): {:.4f}".format(epoch, average_reward))

    # Aktualisiere das Diagramm
    score_plot.append(score)

    # Bildschirm leeren
    screen.fill((0, 0, 0))

    # Zeichne Spieler, Boden, Löcher und den Punktezähler
    pygame.draw.rect(screen, COLORS['WHITE'], [player_x, player_y, player_size, player_size])
    pygame.draw.rect(screen, COLORS['WHITE'], [0, HEIGHT - ground_height, WIDTH, ground_height])
    for hole in holes:
        pygame.draw.rect(screen, COLORS['RED'], hole)
    score_text = font.render("Punkte: {}".format(score), True, COLORS['WHITE'])
    screen.blit(score_text, (10, 10))

    # Überprüfe den Game Over-Status und zeige den entsprechenden Bildschirm an
    if game_over:
        game_over_text = font.render("Game Over! Punkte: {}".format(score), True, COLORS['WHITE'])
        screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 3))
        restart_text = font.render("Restart", True, COLORS['GREEN'])
        exit_text = font.render("Exit", True, COLORS['RED'])
        restart_rect = pygame.Rect(WIDTH // 3, 2 * HEIGHT // 3, restart_text.get_width(), restart_text.get_height())
        exit_rect = pygame.Rect(2 * WIDTH // 3 - exit_text.get_width(), 2 * HEIGHT // 3, exit_text.get_width(), exit_text.get_height())
        pygame.draw.rect(screen, COLORS['GREEN'], restart_rect)
        pygame.draw.rect(screen, COLORS['RED'], exit_rect)
        screen.blit(restart_text, (WIDTH // 3, 2 * HEIGHT // 3))
        screen.blit(exit_text, (2 * WIDTH // 3 - exit_text.get_width(), 2 * HEIGHT // 3))
        restart_game()  # Neustart des Spiels
    else:
        # Aktualisiere den Bildschirm
        pygame.display.flip()

    # Begrenze die Aktualisierungsrate
    clock.tick_busy_loop(FPS)

    # Nach jeder Epoche speichern wir das Modell
    if epoch % 100 == 0:
        save_model(model, model_filepath)
        # Diagramm anzeigen
        plt.plot(score_plot)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Training Score Progress')
        plt.grid(True)
        plt.savefig('score_plot.png')  # Speichern Sie das Diagramm als Bilddatei
        plt.close()


# Nach dem Training das Modell speichern
save_model(model, model_filepath)

# Diagramm anzeigen
plt.plot(score_plot)
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training Score Progress')
plt.grid(True)
plt.savefig('score_plot.png')  # Speichern Sie das Diagramm als Bilddatei
plt.close()
