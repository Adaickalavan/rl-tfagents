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
                
class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space
        
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="tanh")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr))
        #print(self.Actor.summary())

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(self, actions, pred): # for keras custom loss
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.Actor.predict(state)

    def act_continuous(self, state):
        # Use the network to predict the next action to take, using the model
        pred = self.Actor.predict(state)

        low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)

        logp_t = self.gaussian_likelihood(action, pred, self.log_std)

        return action, logp_t

    def gaussian_likelihood(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)

    def act_discrete(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction