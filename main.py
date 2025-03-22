import math
import sys
import time
import pygame
import neat

# Constants
WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 60    
CAR_SIZE_Y = 60
BORDER_COLOR = (255, 255, 255, 255)  # Color to detect collisions
current_generation = 0  # Generation counter

class Car:
    def __init__(self):
        # Load car image from the correct path
        self.sprite = pygame.image.load(r"C:\Users\Personal\Downloads\car.png").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.position = [830, 920]
        self.angle = 0
        self.speed = 20
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.radars = []
        self.alive = True
        self.distance = 0

    def draw(self, screen):
        print("Drawing car at:", self.position)  # Debugging
        screen.blit(self.rotated_sprite, self.position)

    def update(self, game_map):
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        self.alive = game_map.get_at((int(self.center[0]), int(self.center[1]))) != BORDER_COLOR

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

def run_simulation(genomes, config):
    global current_generation
    current_generation += 1
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
    # Load map from the correct path
    game_map = pygame.image.load(r"C:\Users\Personal\Downloads\map2.png").convert()
    
    nets = []
    cars = []
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    clock = pygame.time.Clock()
    start_time = time.time()  # Record the start time

    while time.time() - start_time < 5:  # Run for 5 seconds
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        pygame.display.flip()
        clock.tick(60)  # Control the frame rate

if __name__ == "__main__":
    config_path = r"C:\Users\Personal\Downloads\config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(run_simulation, 1000)
