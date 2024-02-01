import pygame
import random

class Hole:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 0, 0), self.rect)

    @staticmethod
    def generate(x, min_distance, max_size, screen_height):
        if x >= min_distance or x == 0:
            hole_width = random.randint(20, max_size)
            hole_height = random.randint(20, max_size)
            hole_y = screen_height - 20 - hole_height
            return Hole(x, hole_y, hole_width, hole_height)
