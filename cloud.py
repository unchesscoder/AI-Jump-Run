import pygame
import random

class Cloud:
    def __init__(self, x, y, size):
        self.rect = pygame.Rect(x, y, size, size)

    def move(self, speed):
        self.rect.x -= speed // 2

    def draw(self, screen):
        pygame.draw.ellipse(screen, (255, 255, 255), self.rect)
