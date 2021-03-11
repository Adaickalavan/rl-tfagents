# Reinforcement Learning with TF-Agents

## Instructions 
1. Instructions to train a DQN agent in a multi-environment `Breakout-v4` using TF-Agents is given below.

1. The entire code is encapsulated in a single file named `tfagent_dqn.py`.

1. Build the Docker image
    ```bash
    $ cd /path/to/rl-tfagents
    $ docker build --network=host -t rl-tfagents .
    ```

2. Run the container
    ```bash
    # Run the container
    # Note: The source code is mapped from the local host into the 
    # docker container. Change the volume mapping as necessary.
    $ docker run -it --gpus all --network=host --env DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/kyber/workspaces/rl-tfagents/:/src/ rl-tfagents
    # Start RL TFAgent
    $ python3.7 tfagent_dqn.py
    # Start Tensorboard
    $ tensorboard --logdir . --port 6061 &
    ```

## Features
1. The code trains an agent to play `Breakout-v4` environment.

1. Multiple copies of training and evluation environments run in parallel to speed up the data collection (i.e., observations).

## Others
1. Several files implement other stand-alone reinforcement learning algorithms:
+ `policy_gradient.py`: policy gradient algorithm
+ `q_value_iteration.py`: Q-value iteration and Q-value learning
+ `tf_dqn.py`: deep Q-learning in TensorFlow