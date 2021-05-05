import gym
import random
import numpy as np

env = gym.make("BipedalWalker-v3")

def Random_games():
    # Each of this episode is its own game.
    action_size = env.action_space.shape[0]
    for episode in range(10):
        print(f"episode {episode}")
        env.reset()
        # this is each frame, up to 500...but we wont make it that far with random.
        while True:
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()
            
            # This will just create a sample action in any environment.
            # In this environment, the action can be any of one how in list on 4, for example [0 1 0 0]
            action = np.random.uniform(-1.0, 1.0, size=action_size)

            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, and other info.
            next_state, reward, done, info = env.step(action)
            
            # lets print everything in one line:
            #print(reward, action)
            if done:
                break

            # print(f"next state {next_state}")
                
Random_games()