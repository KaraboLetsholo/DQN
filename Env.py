import gym
from DQN import Agent
import numpy as np

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    #env = gym.make("CartPole-v1")
    #print(f"obs dim : {env.observation_space.shape} and act dim {env.action_space.shape}")

    agent = Agent(gamma = 0.99, epsilon=1.0, batchsize=64, n_actions=4,
                   eps_end=0.01, input_dims = [8], lr=0.003)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score+= reward
            agent.store_transition(observation, action, reward, 
                                   new_observation, done)
            agent.learn()
            observation = new_observation
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print("episode", i, "score %.2f" % score,
               "average score %.2f" % avg_score,
               "epsilon %.2f" % agent.epsilon)