import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class CarEnv(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a variant of mountain car with parabolic hilly landscape and 
  additional friction. 
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['console']}
  # Define constants for clearer code
  LEFT = 0
  CENTER = 1
  RIGHT = 2

  def __init__(self, grid_size=10):
    super(CarEnv, self).__init__()

    self.min_position = -3.6
    self.max_position = 1.2
    self.max_speed = 0.07
    self.goal_position = 1
    self.goal_velocity = 0
    
    self.h = 1
    self.gamma = 0.01
    self.force = 0.001
    self.gravity = 0.0025

    self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
    self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
    
    self.viewer = None

    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    self.state = np.array([self.np_random.uniform(low=-0.1, high=-0.1), 0])
    return np.array(self.state)
    
  def _height(self, xs):
    return 1/2*self.h*x^2
  
  def step(self, action):
    """
    Let:
     * H = 1/2*h*x^2
     * theta = tan^{-1}(dH/dx) = tan^{-1}(h*x) 
        => sin(theta) = hx/(h^2x^2+1)^{1/2}
     * dx/dt = v 
     * dv/dt = F - gamma*v - g*sin(theta)
    :return: (np.array) 
    """
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
    
    position, velocity = self.state
    velocity += (action-1)*self.force + (self.h*position)*(-self.gravity)/math.sqrt((self.h*position)**2 + 1) - self.gamma*velocity
    velocity = np.clip(velocity, -self.max_speed, self.max_speed)
    position += velocity
    position = np.clip(position, self.min_position, self.max_position)
    # if (position==self.min_position and velocity<0): velocity = 0

    done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
    reward = -1.0

    self.state = (position, velocity)
    return np.array(self.state), reward, done, {}

  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    # agent is represented as a cross, rest as a dot
    position, velocity = self.state
    print("." * np.abs(position - self.min_position), end="")
    print("x", end="")
    print("." * (self.max_position - position))

  def close(self):
    pass
    
