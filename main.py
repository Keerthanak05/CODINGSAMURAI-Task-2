import math
import sys
import time
import pygame
import neat

# Constants
WIDTH, HEIGHT = 1920, 1080
CAR_SIZE_X, CAR_SIZE_Y = 60, 60
BORDER_COLOR = (255, 255, 255)
current_generation = 0  # Generation counter

class Car:
    def __init__(self):
        self.sprite = pygame.image.load(r"C:\Users\Personal\Downloads\simple car.png").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.position = [830, 920]
        self.angle = 0
        self.speed = 20
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.radars = []
        self.alive = True
        self.distance = 0
    
    def is_collision(self, game_map):
        color = game_map.get_at((int(self.center[0]), int(self.center[1])))
        return sum(abs(color[i] - BORDER_COLOR[i]) for i in range(3)) < 30

    def get_radars(self, game_map):
        self.radars = []
        for angle_offset in [-60, -30, 0, 30, 60]:
            length = 0
            while length < 300:
                x = int(self.center[0] + math.cos(math.radians(self.angle + angle_offset)) * length)
                y = int(self.center[1] - math.sin(math.radians(self.angle + angle_offset)) * length)
                
                if game_map.get_at((x, y)) == BORDER_COLOR:
                    break
                length += 1
            
            self.radars.append(length / 300)  # Normalize distances

    def update(self, game_map):
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]
        self.alive = not self.is_collision(game_map)
        self.get_radars(game_map)

    def get_reward(self):
        return self.distance + sum(self.radars) * 10

def run_simulation(genomes, config):
    global current_generation
    current_generation += 1
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game_map = pygame.image.load(r"C:\Users\Personal\Downloads\road map.png").convert()
    
    nets, cars = [], []
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    clock = pygame.time.Clock()
    start_time = time.time()
    
    while time.time() - start_time < 5:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
        
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                inputs = car.radars
                output = nets[i].activate(inputs)
                
                if output[0] > 0.5:
                    car.angle += 5
                if output[1] > 0.5:
                    car.angle -= 5
                
                genomes[i][1].fitness += car.get_reward()
        
        if still_alive == 0:
            break
        
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)

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
