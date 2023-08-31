# ***************
# AE4350 Final Project
# Chun-Tao Tsao
# Aug 2023
# ***************
import pygame as pg
import random as rd
import numpy as np
import os
import csv

# Initialize pygame
pg.init()

# Constants
s = 400  # set game window size, 400x400 pixels
maxH = 40  # max height of terrain
landP = 25  # landing platform (target) size
plat = 20  # where the platform starts
# Arrays for data ooutput
rewardArr = []
xArr = []
yArr = []
uArr = []
vArr = []
statArr = []

# Generate terrain
mx, my = [], []
for i in range(40):
    mx.append(10 * i)
    my.append(350 - rd.randint(0, maxH))
mx.append(s)
my.append(rd.randint(s - maxH, s))
mx[plat] = mx[20 - 1] + landP
my[plat] = my[20 - 1]
# initialize the previous distance variable for reward system, initialized to the center of platform to origin
previous_distance = np.sqrt(np.square((mx[plat-1] + mx[plat]) / 2) + np.square(my[plat]))

#Main Lunar Lander game
class LunarLanderGame():

    def __init__(self):
        self.s = s
        self.stat = False
        self.y = 10
        self.x = rd.randint(0, s)
        self.u = self.v = 0
        self.g = 1
        self.a = 2
        self.cs = (255, 255, 255)  # Lander initial color is white
        self.ss = ''
        self.agent = LunarLanderQLearningAgent()
        
    # function too call for agent trainiing
    def train(self, episodes=1000, renderstat=False):
        # renderstat determines if the game display needs to be started
        for episode in range(episodes):
            print(f"Starting episode {episode + 1} of {episodes}")
            self.run(renderstat = renderstat)
            # Reset the environment for the next episode
            self.reset_environment()
            
    # Run the game loop
    def run(self, renderstat = True):
        while not self.stat:
            if renderstat:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self.stat = True
            reward = 0
            done = False

            # Agent decides on an action
            action = self.agent.act([self.x, self.y, self.u, self.v], reward, done)
            if action == 0:  # UP
                # Limit the vertical trust so it cannot be <0 (going up)
                if self.v > 0:
                    self.v = max(0, self.v - self.a)
            elif action == 1:  # LEFT
                self.u = self.u - self.a
            elif action == 2:  # RIGHT
                self.u = self.u + self.a
                
            # if the spaceship run out of the side
            if self.x<0:
                self.x = 400 + self.x
            if self.x>400:
                self.x = self.x - 400
            # Update position and velocity of lunar lander
            if self.ss == '':
                self.v += self.g
                self.x = (10 * self.x + self.u) / 10
                self.y = (10 * self.y + self.v) / 10

            # Check for landing or crash
            if mx[plat - 1] <= self.x <= mx[plat] and self.y + 10 == my[plat]: # any where on the buttom tuches the platform successfully
                # Checking for soft landing (i.e., within a vertical speed threshold)
                if abs(self.v) <= 5: 
                    self.ss = 'landed'
                    self.cs = (255, 255, 0) # Color for lander
                else:
                    self.ss = 'crash'
                    # print("landed too fast")
                    self.cs = (255, 0, 0) # Color to indicate a crash (can be adjusted)
            else:
                for i in range(40):
                    if mx[i] <= self.x <= mx[i + 1] and (my[i] <= self.y + 10 or my[i + 1] <= self.y + 10):
                        self.ss = 'crash'
                        self.cs = (255, 0, 0) # Color to indicate a crash (can be adjusted)
                        
            # Determine reward and color of lander based on status
            reward = compute_reward(self.x, self.y, self.u, self.v, self.ss)
                
            # If landed or crashed, reset the agent's state for the next episode
            if self.ss == 'landed' or self.ss == 'crash':
                print("Reward:", reward)
                print(self.ss)
                print("x:", self.x," y:", self.y," u:", self.u, " v:", self.v)
                rewardArr.append(reward)
                xArr.append(self.x)
                yArr.append(self.y)
                uArr.append(self.u)
                vArr.append(self.v)
                statArr.append(self.ss)
                done = True
                self.reset_environment()
                break
            if renderstat:
                # Render game
                screen.fill((0, 0, 0))  # Black background
                # Drawing the terrain
                pg.draw.lines(screen, (255, 255, 255), False, [(mx[i], my[i]) for i in range(41)], 2)  # White terrain line
                # Drawing the landing platform
                pg.draw.line(screen, (0, 0, 255), (mx[plat - 1], my[plat - 1]), (mx[plat], my[plat]), 4)  # Blue landing platform
                pg.draw.rect(screen, self.cs, (self.x, self.y, 10, 10))
                pg.display.flip()
                clock.tick(100)
        pg.quit()
            
    #Reset the environment to its initial state
    def reset_environment(self):
        self.x = rd.randint(0, self.s-10)
        self.y = 10
        self.u = self.v = 0
        self.cs = (255, 255, 255)  # Lander color is reset to white
        self.ss = ''  # Reset status

# Reward calculatiion function
def compute_reward(x, y, u, v, ss):
    global previous_distance
    reward = 0
    #Compute rewards based on agent's state.
    distX = (mx[plat-1] + mx[plat]) / 2 - x
    distY = my[plat] - y
    current_distance = np.sqrt(np.square(distX) + np.square(distY))
    
    # Reward calculation
    if current_distance < previous_distance:
        # The agent has moved closer to the platform
        reward += 1
    else:
        reward -= 1

    # Update the previous distance for the next step
    previous_distance = current_distance

    # Reward or penalty for horizontal direction
    if u * distX > 0:
        reward += 20
    else:
        reward -= 10

    # Consider vertical speed during landing for rewards
    if ss == 'landed' and abs(v) <= 5:  # Safely landed
        reward += 100
    elif ss == 'landed':  # Hard landing
        reward += 50
    elif ss=='crash' and abs(v) <= 5:
        reward += 10 # let the agent know to land softly
    elif ss == 'crash':
        reward -= 50

    return reward

# Q-learning agent for the lunar lander game
class LunarLanderQLearningAgent:
    n_actions = 4

    def __init__(self, alpha=0.5, gamma=0.9, p_explore=0.1, decay_rate=0.995):
        self.n_grid = 150
        self.Q = np.zeros([self.n_grid, self.n_grid, self.n_grid, self.n_grid, self.n_actions])
        self.previous_state = (0, 0, 0, 0)
        self.previous_action = 0
        self.alpha = alpha
        self.gamma = gamma
        self.p_explore = p_explore
        self.decay_rate = decay_rate

    def discretize(self, value, min_value, max_value):
        #Discretize continuous values into bins.#
        step = (max_value - min_value) / self.n_grid
        return min(int((value - min_value) / step), self.n_grid - 1)

    #Decide action based on current observation and learn from reward
    def act(self, observation, reward, done):
        x, y, u, v = observation
        x_d = self.discretize(x, 0, s)
        y_d = self.discretize(y, 0, s)
        u_d = self.discretize(u, -20, 20)
        v_d = self.discretize(v, -20, 20)

        new_state = (x_d, y_d, u_d, v_d)
        
        # Inside act function, after calculating new_state
        # Check and clip indices
        x_d, y_d, u_d, v_d = new_state
        x_d = np.clip(x_d, 0, self.n_grid - 1)
        y_d = np.clip(y_d, 0, self.n_grid - 1)
        u_d = np.clip(u_d, 0, self.n_grid - 1)
        v_d = np.clip(v_d, 0, self.n_grid - 1)

        # Use the clipped state for Q value access
        clipped_state = (x_d, y_d, u_d, v_d)


        # Update Q-values using the Q-learning rule
        self.Q[self.previous_state, self.previous_action] += self.alpha *\
              (reward + self.gamma * max(self.Q[clipped_state]) - self.Q[self.previous_state, self.previous_action])

        # Decaying exploration probability
        if rd.random() < self.p_explore:
            action = rd.randint(0, self.n_actions - 1)
        else:
            action = np.argmax(self.Q[clipped_state])

        self.p_explore *= self.decay_rate

        self.previous_state = new_state
        self.previous_action = action

        return action


# Main Game Loop
screen = pg.display.set_mode((s, s))
pg.display.set_caption('Lunar Lander')
print('platform:', (mx[plat - 1] + mx[plat])/2, my[plat])
clock = pg.time.Clock()
game = LunarLanderGame()
game.train(episodes = 1000, renderstat = False)

episodes = list(range(1, len(xArr) + 1))

# Open a CSV file in write mode
with open('output.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Write the header
    writer.writerow(["Episodes","X", "Y", "U", "V", "Status", "Reward"])

    # Zip the arrays together and write them to the CSV
    for episode, x, y, u, v, status, reward in zip(episodes, xArr,yArr,uArr,vArr,statArr,rewardArr):
        writer.writerow([episode, x, y, u, v, status, reward])
