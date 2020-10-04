import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import random
import pygame

style.use("ggplot")

WIDTH = 300
HEIGHT = 500
BLACK = (0, 0, 0)
LIGHT_GREEN = (32, 212, 0)
DARK_GREEN = (0, 133, 13)
YELLOW = (255, 255, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Flappy Bird")
font = pygame.font.Font('freesansbold.ttf', 32)

# BIRD STUFF
BIRD_INITIAL_X = 100
BIRD_RADIUS = 15
ACCELERATION = -0.5

# OBSTACLE STUFF
MARGIN = 50
CLEARANCE = 100
PIPE_WIDTH = 30
PIPE_INITIAL_X = 250
PIPE_X_DISTANCE = 200
FREE_SPACE = HEIGHT - 2*MARGIN - CLEARANCE
OBSTACLE_SPEED = 2
OBSTACLE_RESTART_X = ((WIDTH % (int(WIDTH/2))) - OBSTACLE_SPEED) + 3*PIPE_X_DISTANCE
SAFE_LINE_DISTANCE = 2*BIRD_RADIUS+5

# AGENT STUFF
EPISODES = 300000
DEATH_PENALTY = -1000
SHOW_EVERY = 200  # how often to play through env visually.
SAVE_EVERY = 50000
start_q_table = "qtable-1588369260.pickle" # None or Filename
LEARNING_RATE = 0.7
DISCOUNT = 1
epsilon = 0.1
epsilon_decay = epsilon/(EPISODES/2)
N_OF_ACTIONS = 2

OBS_HIGH = np.array([PIPE_X_DISTANCE-SAFE_LINE_DISTANCE, HEIGHT-MARGIN, 5.1])
OBS_LOW = np.array([-SAFE_LINE_DISTANCE, -(HEIGHT-MARGIN), -10])

DISCRETE_OS_SIZE = [50, 200, 30]
DISCRETE_OS_SIZE_2 = [x - 1 for x in DISCRETE_OS_SIZE]
discrete_os_win_size = (OBS_HIGH - OBS_LOW) / DISCRETE_OS_SIZE

class Obstacle:
    def __init__(self, x):
        self.x = x
        self.safe_line = self.x+2*BIRD_RADIUS+5
        self.pipes = []
        self.create_rects()

    def create_rects(self):
        rand = random.randint(0,FREE_SPACE)
        self.pipes.append([self.x, 0, self.x+PIPE_WIDTH, MARGIN+rand])
        self.pipes.append([self.x, MARGIN+rand+CLEARANCE, self.x+PIPE_WIDTH, HEIGHT])

    def move(self):
        self.x -= OBSTACLE_SPEED
        self.safe_line = self.x+2*BIRD_RADIUS+5

        if(self.x) < 0:
            self.x = OBSTACLE_RESTART_X
            self.safe_line = self.x+SAFE_LINE_DISTANCE
            rand = random.randint(0,FREE_SPACE)
            self.pipes[0] = [self.x, 0, self.x+PIPE_WIDTH, MARGIN+rand]
            self.pipes[1] = [self.x, MARGIN+rand+CLEARANCE, self.x+PIPE_WIDTH, HEIGHT]

        for pipe in self.pipes:
            pipe[0] = self.x
            pipe[2] = self.x+PIPE_WIDTH

class Bird:
    def __init__(self):
        self.x = BIRD_INITIAL_X
        self.y = 2 + np.random.randint(0, HEIGHT-2*BIRD_RADIUS-2)
        self.v = 0
        self.dim = 2*BIRD_RADIUS
        self.rect = [self.x, self.y, self.x+self.dim, self.y+self.dim]

    def action(self, choice):
        if choice == 0:
            self.move(False)
        elif choice == 1:
            self.move(True)
        else:
            print("not a valid choice")

    def move(self, jumping):
        if(jumping):
            self.v = 5
        else:
            self.v += ACCELERATION
        self.y -= self.v
        self.rect = [self.x, self.y, self.x+self.dim, self.y+self.dim]

    def intersects(self, rect):
        if(self.rect[0] > rect[2] or rect[0] > self.rect[2]):
            return False
        if(self.rect[1] > rect[3] or rect[1] > self.rect[3]):
            return False
        return True

    def crosses_x(self, x_cord):
        if(self.rect[0]>x_cord):
            return True
        return False

    def crosses_y(self):
        if(self.rect[1] < 0 or self.rect[3] > HEIGHT):
            return True
        return False


if start_q_table is None:
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [N_OF_ACTIONS]))
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

def get_discrete_state(state):
    discrete_state = (state - OBS_LOW) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

def get_obs(bird, obstacle):
    hor_distance = obstacle.x - bird.x
    ver_distance = bird.rect[1] - obstacle.pipes[0][3]
    return (hor_distance, ver_distance, bird.v)

scores_dict = {"ep": [], "score": []}
scores = []

for episode in range(EPISODES):
    bird = Bird()
    obstacles = []
    for i in range(3):
        obstacles.append(Obstacle(PIPE_INITIAL_X+i*PIPE_X_DISTANCE))
    next_obstacle = 0
    if episode % SHOW_EVERY == 0:
        show = True
    else:
        show = False

    done = False
    score = 0
    while not done:
        reward = 0
        obs = get_discrete_state(get_obs(bird, obstacles[next_obstacle]))
        if(np.random.uniform()>epsilon):
            action = np.argmax(q_table[obs])
        else:
            action = 0

        # Take the action!
        bird.action(action)

        for obstacle in obstacles:
            obstacle.move()

        if(bird.crosses_x(obstacles[next_obstacle].safe_line)):
            score += 1
            if next_obstacle == 2:
                next_obstacle = 0
            else:
                next_obstacle += 1

        for rect in obstacles[next_obstacle].pipes:
            if(bird.intersects(rect)):
                done = True
                reward = DEATH_PENALTY

        if(bird.crosses_y()):
            done = True
            reward = DEATH_PENALTY

        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = get_obs(bird, obstacles[next_obstacle])
        new_discrete_obs = get_discrete_state(new_obs)
        max_future_q = np.max(q_table[new_discrete_obs])
        current_q = q_table[new_discrete_obs][action]

        if reward == DEATH_PENALTY:
            new_q = DEATH_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            screen.fill(LIGHT_GREEN)
            for obstacle in obstacles:
                pygame.draw.rect(screen, DARK_GREEN, [obstacle.pipes[0][0], obstacle.pipes[0][1], obstacle.pipes[0][2]-obstacle.pipes[0][0], obstacle.pipes[0][3]-obstacle.pipes[0][1]])
                pygame.draw.rect(screen, DARK_GREEN, [obstacle.pipes[1][0], obstacle.pipes[1][1], obstacle.pipes[1][2]-obstacle.pipes[1][0], obstacle.pipes[1][3]-obstacle.pipes[1][1]])
            pygame.draw.circle(screen, YELLOW, (bird.x+BIRD_RADIUS, bird.y+BIRD_RADIUS), BIRD_RADIUS)
            text = font.render(str(score), True, BLACK)
            textRect = text.get_rect()
            textRect.center = (WIDTH / 2, 30)
            screen.blit(text, textRect)
            #pygame.draw.line(screen, BLACK, (bird.x, bird.y), (bird.x+new_obs[0], bird.y))
            #pygame.draw.line(screen, BLACK, (bird.x, bird.y), (bird.x, bird.y-new_obs[1]))
            pygame.display.flip()
            time.sleep(0.02)

    if(epsilon>0):
        epsilon -= epsilon_decay
    scores.append(score)
    scores_dict["ep"].append(episode)
    scores_dict["score"].append(score)


    if(show):
        print(epsilon_decay)
        print(f"episode: {episode}, epsilon: {epsilon}, mean: {np.mean(scores[-SHOW_EVERY:])}")
    if episode % SAVE_EVERY == 0:
        with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
            pickle.dump(q_table, f)

pygame.quit()

fig = plt.scatter(scores_dict["ep"], scores_dict["score"])
plt.title("Results")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
