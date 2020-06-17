# Importing libs
import numpy as np
import os
import time
import torch
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.logger import Logger
from PIL import Image as PILImage
from models import ReplayBuffer, TD3
from scipy.ndimage import rotate

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

seed = 0 # for uniformity
torch.manual_seed(seed)
np.random.seed(seed)
save_models = True #flag for saving models

file_name = "%s_%s_%s" % ("TD3", "DDPG", str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")
if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

#To keep the last point in memory when we draw the outRoad on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
last_reward = 0
# Starting point of truck in training
origin_x = 144
origin_y = 216
scores = []
im = CoreImage("./images/MASK1.png")

def init():
    global outRoad
    global goal_x
    global goal_y
    global first_update
    global outRoadCount
    # Loading road map
    outRoad = np.zeros((longueur, largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    outRoad = np.asarray(img) / 255
    #Chossing random mail point
    goal_x, goal_y = coordinates[np.random.randint(0, 13)]
    first_update = False

class Truck(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    cropsize = 28
    padsize = 28
    view = np.zeros([1,int(cropsize),int(cropsize)])

    def turn(self, rotation):
        global episode_num
        global padsize
        global cropsize
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        # Cropping section of road from map and superimposing truck with its orientation on it to train for action
        outRoadVar = np.copy(outRoad)
        outRoadVar = np.pad(outRoadVar,self.padsize,constant_values=1.0)
        outRoadVar = outRoadVar[int(self.x) - self.cropsize + self.padsize:int(self.x) + self.cropsize + self.padsize,
                   int(self.y) - self.cropsize + self.padsize:int(self.y) + self.cropsize + self.padsize]
        outRoadVar = rotate(outRoadVar, angle=90-(self.angle-90), reshape= False,
                                        order=1, mode='constant',  cval=1.0)
        outRoadVar[int(self.padsize)-5:int(self.padsize),
                            int(self.padsize) - 2:int(self.padsize) + 3 ] = 0.6
        outRoadVar[int(self.padsize):int(self.padsize) + 5,
                            int(self.padsize) - 2:int(self.padsize) + 3] = 0.3
        self.view=outRoadVar
        self.view = self.view[::2, ::2]
        self.view = np.expand_dims(self.view, 0)


class Mail(Widget):
    pass
class Box(Widget):
    pass

#List of fixed Mail/Box points to choose from randomly
coordinates = [[705,437],[144,216],[163,429],[436,482],[448,174],[645,87],[879,206],[829,458],[976,518],[1057,273],[1120,206],[1170,479],[226,77]]
first_update = True # Setting the first mail as goal for training start
last_distance = 0

class Game(Widget):
    truck = ObjectProperty(None)
    mail = ObjectProperty(None)
    box = ObjectProperty(None)

    def serve_truck(self):
        self.truck.center = self.center
        self.truck.velocity = Vector(3, 0)

    def reset(self):
        global last_distance
        global origin_x
        global origin_y
        self.truck.x = origin_x
        self.truck.y = origin_y
        xx = goal_x - self.truck.x
        yy = goal_y - self.truck.y
        orientation = Vector(*self.truck.velocity).angle((xx,yy))/180.
        self.distance = np.sqrt((self.truck.x - goal_x) ** 2 + (self.truck.y - goal_y) ** 2)
        state = [self.truck.view, orientation, -orientation, last_distance - self.distance]
        return state


    def step(self,action):
        global goal_x
        global goal_y
        global origin_x
        global origin_y
        global done
        global last_distance
        global outRoadCount
        global distance_travelled

        rotation = action.item()
        self.truck.turn(rotation)
        self.distance = np.sqrt((self.truck.x - goal_x) ** 2 + (self.truck.y - goal_y) ** 2)
        xx = goal_x - self.truck.x
        yy = goal_y - self.truck.y
        orientation = Vector(*self.truck.velocity).angle((xx, yy)) / 180.
        state = [self.truck.view, orientation, -orientation, last_distance-self.distance]

        #When going out of the road
        if outRoad[int(self.truck.x), int(self.truck.y)] > 0:
            self.truck.velocity = Vector(0.5, 0).rotate(self.truck.angle)
            last_reward = -7.0

        # When moving on road
        else:
            self.truck.velocity = Vector(1.5, 0).rotate(self.truck.angle)
            last_reward = -1.7
            #When reducing destination distance
            if self.distance < last_distance:
                last_reward = 0.6
        # egde conditions for truck
        if self.truck.x < 5:
            self.truck.x = 5
            last_reward = -15
        if self.truck.x > self.width - 5:
            self.truck.x = self.width - 5
            last_reward = -15
        if self.truck.y < 5:
            self.truck.y = 5
            last_reward = -15
        if self.truck.y > self.height - 5:
            self.truck.y = self.height - 5
            last_reward = -15

        #When achieved goal
        if self.distance < 30:
            if self.tomail == 1:
                origin_x = goal_x
                origin_y = goal_y
                goal_x,goal_y= coordinates[np.random.randint(0,13)]
                self.box.x = goal_x
                self.box.y = goal_y
                self.tomail = 0
                last_reward = 110
                self.mail.size = 15,15
                self.box.size = 40, 40
                done = True
            else:
                origin_x = goal_x
                origin_y = goal_y
                goal_x,goal_y= coordinates[np.random.randint(0,13)]
                self.mail.x = goal_x
                self.mail.y = goal_y
                self.tomail = 1
                last_reward = 110
                self.mail.size = 20, 20
                self.Box.size = 1, 1
                done = True
        if self.tomail == 0:
            self.mail.x = self.truck.x - 5
            self.mail.y = self.truck.y - 5

        last_distance = self.distance
        return state, last_reward, done

    def evaluate_policy(self, policy, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            # Resetting environment
            obs = self.reset()
            done = False
            while not done:
                action = policy.select_action(np.array(obs))
                obs,reward,done = self.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print("---------------------------------------")
        return avg_reward

    def update(self, dt):
        global scores
        global first_update
        global goal_x
        global goal_y
        global longueur
        global largeur
        global last_reward

        global policy
        global done
        global episode_reward
        global replay_buffer
        global obs
        global new_obs
        global evaluations

        global episode_num
        global total_timesteps
        global timesteps_since_eval
        global max_timesteps
        global max_episode_steps
        global episode_timesteps
        global distance_travelled
        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            self.tomail=1
            self.mail.x = goal_x
            self.mail.y = goal_y
            evaluations = [self.evaluate_policy(policy)]
            distance_travelled=0
            done = True
            obs = self.reset()
        if episode_reward<-3000:
            done=True
        if total_timesteps < max_timesteps:
            #when episode is finished
            if done:
                if total_timesteps != 0:
                    Logger.info("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num,
                                                                                  episode_reward))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip,
                                 policy_freq)
                #evaluating episode and saving policy
                if timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    evaluations.append(self.evaluate_policy(policy))
                    policy.save(file_name, directory="./pytorch_models")
                    np.save("./results/%s" % (file_name), evaluations)
                #Resetting environment after episode
                obs = self.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
            #Random actions
            if total_timesteps < start_timesteps:
                action = np.random.uniform(low=-3, high=3, size=(1,))
            # using Model
            else:
                action = policy.select_action(np.array(obs))
                # adding exploration noise and clipping
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=1)).clip(-3, 3)
            #Executing action, moving to next state, collecting reward
            new_obs,reward, done = self.step(action)
            done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(
                done)
            #increasing reward
            episode_reward += reward
            # storing new transitions in replay_buffer
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            if total_timesteps%10==1:
                Logger.info(" ".join([str(total_timesteps), str(obs[1:]), str(new_obs[1:]), str(action), str(reward), str(done_bool)]))
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
            if total_timesteps%5000==1:
                #Saving model
                Logger.info("Saving Model %s" % (file_name))
                policy.save("%s" % (file_name), directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)
        else:
            action = policy.select_action(np.array(obs))
            new_obs,reward, done = self.step(action)
            obs = new_obs
            total_timesteps += 1
            if total_timesteps%1000==1:
                print(total_timesteps)


class TruckApp(App):
    def build(self):
        parent = Game()
        parent.serve_truck()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        return parent

#Initializing params
start_timesteps = 2e3 #Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 1e3 #How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e4 #Total number of iterations/timesteps
expl_noise = 0.08 # Exploration noise
batch_size = 200 # Size of the batch
discount = 0.99 # Discount factor gamma
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise
noise_clip = 0.5 # Maximum Gaussian noise
policy_freq = 2
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
episode_reward=0
t0 = time.time()
distance_travelled=0
max_episode_steps = 1000
done = True
load_model=False # for training

state_dim = 4
action_dim = 1
max_action = 3

replay_buffer = ReplayBuffer()
policy = TD3(state_dim, action_dim, max_action)

obs=np.array([])
new_obs=np.array([])
evaluations=[]
#Loading trained model
if load_model == True:
    total_timesteps = max_timesteps
    policy.load("%s" % (file_name), directory="./pytorch_models")

TruckApp().run()
