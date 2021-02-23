#Implementing TF-Agent DQN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import logging
import numpy as np
import pathlib
import tensorflow as tf
import tf_agents
import time

from tensorflow import keras
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import suite_gym, suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import function, Checkpointer

def main(_):
    # Environment
    env_name = "Breakout-v4"
    train_num_parallel_environments=4
    max_steps_per_episode = 1000
    # Replay buffer
    replay_buffer_capacity = 50000
    init_replay_buffer = 500
    # Driver
    collect_steps_per_iteration = 1 * train_num_parallel_environments
    # Training
    train_batch_size = 32
    train_iterations = 100000
    train_summary_interval=200
    train_checkpoint_interval=200
    # Evaluation
    eval_num_parallel_environments=1
    eval_summary_interval=500
    eval_num_episodes=20
    # File paths
    path = pathlib.Path(__file__)
    parent_dir = path.parent.resolve()
    folder_name = path.stem + time.strftime("_%Y%m%d_%H%M%S")
    train_checkpoint_dir = str(parent_dir / folder_name / "train_checkpoint")
    train_summary_dir = str(parent_dir / folder_name / "train_summary")
    eval_summary_dir = str(parent_dir / folder_name / "eval_summary")

    # Parallel training environment
    tf_env = TFPyEnvironment(
                ParallelPyEnvironment([
                        lambda: suite_atari.load(
                            env_name,
                            env_wrappers=[
                                lambda env: TimeLimit(env, duration=max_steps_per_episode)
                            ],
                            gym_env_wrappers=[
                                AtariPreprocessing, FrameStack4
                            ],
                        )
                    ]*train_num_parallel_environments
                )
            )
    tf_env.seed([42]*tf_env.batch_size)
    tf_env.reset()

    # Parallel evaluation environment
    eval_tf_env = TFPyEnvironment(
                    ParallelPyEnvironment([
                            lambda: suite_atari.load(
                                env_name,
                                env_wrappers=[
                                    lambda env: TimeLimit(env, duration=max_steps_per_episode)
                                ],
                                gym_env_wrappers=[
                                    AtariPreprocessing, FrameStack4
                                ],
                            )
                        ]*eval_num_parallel_environments
                    )
                )
    eval_tf_env.seed([42]*eval_tf_env.batch_size)
    eval_tf_env.reset()

    # Creating the Deep Q-Network
    preprocessing_layer = keras.layers.Lambda(
            lambda obs: tf.cast(obs, np.float32) / 255.
        )

    conv_layer_params=[
        (32, (8, 8), 4), 
        (64, (4, 4), 2), 
        (64, (3, 3), 1)]
    fc_layer_params=[512]

    q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params)

    # Creating the DQN Agent
    optimizer = keras.optimizers.RMSprop(
                    lr=2.5e-4, rho=0.95, momentum=0.0,
                    epsilon=0.00001, centered=True)

    epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0, # initial ε
        decay_steps=2500000,
        end_learning_rate=0.01) # final ε

    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=200,
        td_errors_loss_fn=keras.losses.Huber(reduction="none"),
        gamma=0.99, # discount factor
        train_step_counter=global_step,
        epsilon_greedy=lambda: epsilon_fn(global_step))
    agent.initialize()

    # Creating the Replay Buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    # Observer: Replay Buffer Observer
    replay_buffer_observer = replay_buffer.add_batch

    # Observer: Training Metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=tf_env.batch_size),
        ]

    # Creating the Collect Driver
    collect_driver = DynamicStepDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer_observer] + train_metrics,
        num_steps=collect_steps_per_iteration)

    # Initialize replay buffer
    initial_collect_policy = RandomTFPolicy(
                                tf_env.time_step_spec(),
                                tf_env.action_spec())
    init_driver = DynamicStepDriver(
                    tf_env,
                    initial_collect_policy,
                    observers=[replay_buffer_observer, ShowProgress(init_replay_buffer)],
                    num_steps=init_replay_buffer)
    final_time_step, final_policy_state = init_driver.run()

    # Creating the Dataset
    dataset = replay_buffer.as_dataset(
        sample_batch_size=train_batch_size,
        num_steps=2,
        num_parallel_calls=3).prefetch(3)

    # Optimize by wrapping some of the code in a graph using TF function.
    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)

    print("\n\n++++++++++++++++++++++++++++++++++\n")

    # Create checkpoint
    train_checkpointer = Checkpointer(
        ckpt_dir=train_checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        # replay_buffer=replay_buffer,
        global_step=global_step,
        # metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics')
        )

    # Restore checkpoint
    # train_checkpointer.initialize_or_restore()

    # Summary writers and metrics
    train_summary_writer = tf.summary.create_file_writer(train_summary_dir)
    eval_summary_writer = tf.summary.create_file_writer(eval_summary_dir)
    eval_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=eval_tf_env.batch_size, buffer_size=eval_num_episodes),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=eval_tf_env.batch_size, buffer_size=eval_num_episodes)
    ]

    # Create evaluate callback function
    eval_callback = evaluate(
        eval_metrics=eval_metrics,
        eval_tf_env=eval_tf_env,
        eval_policy=agent.policy,
        eval_num_episodes=eval_num_episodes,
        train_step=global_step,
        eval_summary_writer=eval_summary_writer)

    # Train agent
    train_agent(
        tf_env=tf_env, 
        train_iterations=train_iterations, 
        global_step=global_step, 
        agent=agent, 
        dataset=dataset, 
        collect_driver=collect_driver,
        train_metrics=train_metrics, 
        train_checkpointer=train_checkpointer, 
        train_checkpoint_interval=train_checkpoint_interval, 
        train_summary_writer=train_summary_writer, 
        train_summary_interval=train_summary_interval,
        eval_summary_interval=eval_summary_interval,
        eval_callback=eval_callback)
    
    print("\n\n++++++++++ END OF TF_AGENTS RL TRAINING ++++++++++\n\n")

def train_agent(
    tf_env, 
    train_iterations, 
    global_step, 
    agent, 
    dataset, 
    collect_driver, 
    train_metrics, 
    train_checkpointer, 
    train_checkpoint_interval, 
    train_summary_writer, 
    train_summary_interval,
    eval_summary_interval,
    eval_callback):

    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    timed_at_step = global_step.numpy()
    time_acc = 0
    train_summary_writer.set_as_default()
    for iteration in range(train_iterations):
        # Start timer
        start_time = time.time()

        # Collect a few steps using collect_policy and save to the replay buffer.
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        
        # Sample a batch of data from the buffer and update the agent's network.
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)

        # Stop timer
        time_acc += time.time() - start_time

        # Checkpoint
        if global_step.numpy() % train_checkpoint_interval == 0:
            train_checkpointer.save(global_step=global_step.numpy())
            print("\n")

        # Print training metrics
        if iteration % 100 == 0:
            print(f"\rTraining iteration: {agent.train_step_counter.numpy()}, Loss:{train_loss.loss.numpy():.5f}", end="")
            metric_utils.log_metrics(train_metrics)
            print("\n")

        # Summary writer
        with tf.summary.record_if(lambda: tf.math.equal(global_step % train_summary_interval, 0)):
            # Training iteration speed (i.e., training frames per second)
            steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
            tf.summary.scalar(name='global_steps_per_sec', data=steps_per_sec, step=global_step)
            timed_at_step = global_step.numpy()
            time_acc = 0
            # Write training metrics to summary
            for train_metric in train_metrics:
                train_metric.tf_summaries(train_step=global_step, step_metrics=train_metrics[:2])

        # Evaluate the learned policy and network
        if global_step.numpy() % eval_summary_interval == 0 and global_step.numpy() > 0:
            print("Evaluating learned policy")
            eval_callback()
            print("\n")

    print("\n")

# Evaluation
def evaluate(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        eval_num_episodes,
        train_step,
        eval_summary_writer):

    def compute():
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            eval_num_episodes,
            train_step,
            eval_summary_writer,
            'Metrics - Evaluation'
        )
        # result = metric_utils.compute(
        #     eval_metrics, 
        #     eval_tf_env,
        #     eval_policy, 
        #     eval_num_episodes
        # )
        metric_utils.log_metrics(eval_metrics)

    return compute

# Observer: Show progress
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary()[0]:
            self.counter += 1 
        if self.counter % 100 == 0:
            print(f"\rInitialize replay buffer: {self.counter}/{self.total}", end="")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # Suppress tensorflow deprecation warning messages 
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    tf.random.set_seed(42)
    np.random.seed(42)

    tf_agents.system.multiprocessing.handle_main(main)
