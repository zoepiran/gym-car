import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='car-v0',
    entry_point='gym_car.envs:CarEnv',
    max_episode_steps=200,
)
