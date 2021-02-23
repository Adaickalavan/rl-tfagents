import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from functools import partial
from itertools import cycle
from pathlib import Path
import random
from time import time
from typing import Tuple, List, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import tensorflow as tf
from tf_agents.agents import DqnAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs import TensorSpec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import common

print('Physical Devices:\n', tf.config.list_physical_devices(), '\n\n')

OUTPUTS_DIR = f'./outputs/{int(10000000 * time())}'
print('Output Directory:', OUTPUTS_DIR)


class IMAgent(DqnAgent):
    def __init__(self,
                 env: TFPyEnvironment,
                 observation_spec: TensorSpec = None,
                 action_spec: TensorSpec = None,
                 reward_fn: Callable = lambda time_step: time_step.reward,
                 action_fn: Callable = lambda action: action,
                 name: str='IMAgent',
                 q_network=None,
                 # training params
                 replay_buffer_max_length: int = 1000,
                 learning_rate: float = 1e-5,
                 training_batch_size: int = 8,
                 training_parallel_calls: int = 3,
                 training_prefetch_buffer_size: int = 3,
                 training_num_steps: int = 2,
                 **dqn_kwargs):

        self._env = env
        self._reward_fn = reward_fn
        self._name = name
        self._observation_spec = observation_spec or self._env.observation_spec()
        self._action_spec = action_spec or self._env.action_spec()
        self._action_fn = action_fn

        q_network = q_network or self._build_q_net()

        env_ts_spec = self._env.time_step_spec()
        time_step_spec = TimeStep(
            step_type=env_ts_spec.step_type,
            reward=env_ts_spec.reward,
            discount=env_ts_spec.discount,
            observation=q_network.input_tensor_spec
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        super().__init__(time_step_spec,
                         self._action_spec,
                         q_network,
                         optimizer,
                         name=name,
                         **dqn_kwargs)

        self._policy_state = self.policy.get_initial_state(
            batch_size=self._env.batch_size)
        self._rewards = []

        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=self.collect_data_spec,
            batch_size=self._env.batch_size,
            max_length=replay_buffer_max_length)
        
        self._training_batch_size = training_batch_size
        self._training_prefetch_buffer_size = training_prefetch_buffer_size
        self._training_num_steps = training_num_steps
        self.train = common.function(self.train)

    def _build_q_net(self):
        fc_layer_params = (50,)

        q_net = QNetwork(
            self._observation_spec,
            self._action_spec,
            fc_layer_params=fc_layer_params)

        q_net.create_variables()
        q_net.summary()
        
        return q_net

    def reset(self):
        self._policy_state = self.policy.get_initial_state(
            batch_size=self._env.batch_size
        )
        self._rewards = []
        
    def episode_return(self) -> float:
        return np.sum(self._rewards)
        
    def _observation_fn(self, observation: tf.Tensor) -> tf.Tensor:
        """
            Takes a tensor with specification self._env.observation_spec
            and extracts a tensor with specification self._observation_spec.
            
            For example, consider an agent within an NxN maze environment. 
            The env could expose the entire NxN integer matrix as an observation
            but we would prefer the agent to only see a 3x3 window around their
            current location. To do this we can override this method.
            
            This allows us to have different agents acting in the same environment
            with different observations.
        """
        return observation

    def _augment_time_step(self, time_step: TimeStep) -> TimeStep:

        reward = self._reward_fn(time_step)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        if reward.shape != time_step.reward.shape:
            reward = tf.reshape(reward, time_step.reward.shape)
            
        observation = self._observation_fn(time_step.observation)

        return TimeStep(
            step_type=time_step.step_type,
            reward=reward,
            discount=time_step.discount,
            observation=observation
        )

    def _current_time_step(self) -> TimeStep:
        time_step = self._env.current_time_step()
        time_step = self._augment_time_step(time_step)
        return time_step

    def _step_environment(self, action) -> TimeStep:
        action = self._action_fn(action)
        time_step = self._env.step(action)
        time_step = self._augment_time_step(time_step)
        return time_step

    def act(self, collect=False) -> Trajectory:
        time_step = self._current_time_step()

        if collect:
            policy_step = self.collect_policy.action(
                time_step, policy_state=self._policy_state)
        else:
            policy_step = self.policy.action(
                time_step, policy_state=self._policy_state)

        self._policy_state = policy_step.state
        next_time_step = self._step_environment(policy_step.action)
        traj = trajectory.from_transition(time_step, policy_step, next_time_step)

        self._rewards.append(next_time_step.reward)
        
        if collect:
            self._replay_buffer.add_batch(traj)

        return traj

    def train_iteration(self) -> LossInfo:
        experience, info = self._replay_buffer.get_next(
            sample_batch_size=self._training_batch_size,
            num_steps=self._training_num_steps
        )
        return self.train(experience)


from tf_agents.environments.examples.tic_tac_toe_environment import TicTacToeEnvironment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType

REWARD_ILLEGAL_MOVE = np.asarray(-5, dtype=np.float32)

class TicTacToeMultiAgentEnv(TicTacToeEnvironment):
    
    def action_spec(self):
        position_spec = BoundedArraySpec((1,), np.int32, minimum=0, maximum=8)
        value_spec = BoundedArraySpec((1,), np.int32, minimum=1, maximum=2)
        return {
            'position': position_spec,
            'value': value_spec
        }
    
    def _step(self, action: np.ndarray):
        if self._current_time_step.is_last():
            return self._reset()

        index_flat = np.array(range(9)) == action['position']
        index = index_flat.reshape(self._states.shape) == True
        if self._states[index] != 0:
            return TimeStep(StepType.LAST, 
                            REWARD_ILLEGAL_MOVE,
                            self._discount, 
                            self._states)

        self._states[index] = action['value']

        is_final, reward = self._check_states(self._states)
        
        if np.all(self._states == 0):
            step_type = StepType.FIRST
        elif is_final:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        return TimeStep(step_type, reward, self._discount, self._states)

def print_tic_tac_toe(state):
    table_str = '''
    {} | {} | {}
    - + - + -
    {} | {} | {}
    - + - + -
    {} | {} | {}
    '''.format(*tuple(state.flatten()))
    table_str = table_str.replace('0', ' ')
    table_str = table_str.replace('1', 'X')
    table_str = table_str.replace('2', 'O')
    print(table_str)

def ttt_action_fn(player, action):
    return {'position': action, 'value': player}

tf_ttt_env = TFPyEnvironment(tic_tac_toe_env)

player_1 = IMAgent(
    tf_ttt_env,
    action_spec = tf_ttt_env.action_spec()['position'],
    action_fn = partial(ttt_action_fn, 1),
    name='Player1'
)

def p2_reward_fn(ts: TimeStep) -> float:
    if ts.reward == -1.0:
        return 1.0
    if ts.reward == 1.0:
        return -1.0
    return ts.reward

player_2 = IMAgent(
    tf_ttt_env,
    action_spec = tf_ttt_env.action_spec()['position'],
    action_fn = partial(ttt_action_fn, 2),
    reward_fn = p2_reward_fn,
    name='Player2'
)