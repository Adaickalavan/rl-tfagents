# Reinforcement Learning with TF Agents

## Instructions to run the code
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
    # Start Tensorboard
    $ tensorboard --logdir . --port 6061 &
    # Start RL TFAgent
    $ python3.7 tfagent_dqn.py
    ```

## Features
1. The code trains an agent to play `Breakout-v4` environment.

1. Multiple copies of training and evluation environments run in parallel to speed up the data collection (i.e., observations).
