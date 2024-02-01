import pygame
import sys
import random
from player import Player
from ground import Ground
from hole import Hole
from cloud import Cloud

class JumpAndRunGame:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.FPS = 60
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Jump and Run Game")
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()
        self.player = Player(self.WIDTH // 4, self.HEIGHT - 100, 50, 8)
        self.ground = Ground(20, self.WIDTH)
        self.holes = []
        self.clouds = []
        self.score = 0
        self.game_over = False
        self.restart_rect = pygame.Rect(0, 0, 0, 0)
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        self.min_hole_distance = 200
        self.hole_frequency = 20
        self.max_hole_size = 100
        self.cloud_frequency = 5

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and self.game_over:
                mouse_pos = pygame.mouse.get_pos()
                if self.is_mouse_on_button(mouse_pos, self.restart_rect):
                    self.reset_game()
                elif self.is_mouse_on_button(mouse_pos, self.exit_rect):
                    pygame.quit()
                    sys.exit()

        keys = pygame.key.get_pressed()
        if not self.player.is_jumping and not self.game_over:
            if keys[pygame.K_SPACE]:
                self.player.is_jumping = True

    def reset_game(self):
        self.player.y = self.HEIGHT - self.player.size * 2
        self.holes = []
        self.score = 0
        self.game_over = False

    def is_mouse_on_button(self, mouse_pos, button_rect):
        return button_rect.collidepoint(mouse_pos)

    def update_holes(self):
        for hole in self.holes:
            hole.rect.x -= self.player.speed

        if random.randint(0, self.hole_frequency) == 0:
            new_hole = Hole.generate(self.WIDTH, self.min_hole_distance, self.max_hole_size, self.HEIGHT)
            if new_hole:
                self.holes.append(new_hole)
                self.score += 1

        self.holes = [hole for hole in self.holes if hole.rect.x + hole.rect.width > 0]

    def update_clouds(self):
        for cloud in self.clouds:
            cloud.move(self.player.speed)

        if random.randint(0, 100) < self.cloud_frequency:
            self.clouds.append(Cloud(self.WIDTH, random.randint(50, 200), random.randint(20, 50)))

        self.clouds = [cloud for cloud in self.clouds if cloud.rect.x + cloud.rect.width > 0]

    def draw_game_over(self):
        game_over_text = self.font.render(f"Game Over! Punkte: {self.score}", True, (255, 255, 255))
        self.screen.blit(game_over_text, (self.WIDTH // 2 - game_over_text.get_width() // 2, self.HEIGHT // 3))

        restart_text = self.font.render("Restart", True, (255, 255, 255))
        exit_text = self.font.render("Exit", True, (255, 255, 255))

        self.restart_rect = pygame.Rect(self.WIDTH // 3, 2 * self.HEIGHT // 3, restart_text.get_width(),
                                        restart_text.get_height())
        self.exit_rect = pygame.Rect(2 * self.WIDTH // 3 - exit_text.get_width(), 2 * self.HEIGHT // 3,
                                     exit_text.get_width(), exit_text.get_height())

        pygame.draw.rect(self.screen, (0, 128, 0), self.restart_rect)
        pygame.draw.rect(self.screen, (128, 0, 0), self.exit_rect)

        self.screen.blit(restart_text, (self.WIDTH // 3, 2 * self.HEIGHT // 3))
        self.screen.blit(exit_text, (2 * self.WIDTH // 3 - exit_text.get_width(), 2 * self.HEIGHT // 3))

    def main_loop(self):
        while True:
            self.handle_events()

            if not self.player.is_jumping and not self.game_over:
                self.player.jump()

            self.update_holes()
            self.update_clouds()

            if not self.game_over:
                for hole in self.holes:
                    if hole.rect.colliderect(
                            pygame.Rect(self.player.x, self.player.y, self.player.size, self.player.size)):
                        self.game_over = True

            self.screen.fill((135, 206, 250))  # Sky Color

            for cloud in self.clouds:
                cloud.draw(self.screen)

            self.player.draw(self.screen)
            self.ground.draw(self.screen)

            for hole in self.holes:
                hole.draw(self.screen)

            score_text = self.font.render(f"Punkte: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))

            if self.game_over:
                self.draw_game_over()

            pygame.display.flip()
            self.clock.tick(self.FPS)

if __name__ == "__main__":
    game = JumpAndRunGame()
    game.main_loop()
