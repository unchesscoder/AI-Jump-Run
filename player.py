import pygame

class Player:
    def __init__(self, x, y, size, speed):
        self.x = x
        self.y = y
        self.size = size
        self.speed = speed
        self.is_jumping = False
        self.jump_count = 10

    def jump(self):
        if self.jump_count >= -10:
            neg = 1 if self.jump_count >= 0 else -1
            self.y -= (self.jump_count ** 2) * 0.5 * neg
            self.jump_count -= 1
        else:
            self.is_jumping = False
            self.jump_count = 10

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (self.x + self.size // 2, self.y), self.size // 4)
        pygame.draw.line(screen, (255, 255, 255), (self.x + self.size // 2, self.y + self.size // 4),
                         (self.x + self.size // 2, self.y + self.size // 2 + self.size // 4), 2)
        pygame.draw.line(screen, (255, 255, 255), (self.x, self.y + self.size // 4 + self.size // 8),
                         (self.x + self.size, self.y + self.size // 4 + self.size // 8), 2)
        pygame.draw.line(screen, (255, 255, 255), (self.x + self.size // 2, self.y + self.size // 2 + self.size // 4),
                         (self.x, self.y + self.size), 2)
        pygame.draw.line(screen, (255, 255, 255), (self.x + self.size // 2, self.y + self.size // 2 + self.size // 4),
                         (self.x + self.size, self.y + self.size), 2)

