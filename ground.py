import pygame

class Ground:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), [0, screen.get_height() - self.height, self.width, self.height])
