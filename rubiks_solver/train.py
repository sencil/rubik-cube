from cube_env import RubiksCubeEnv
from dqn_agent import DQNAgent
import numpy as np

# Initialize environment and agent
env = RubiksCubeEnv()
agent = DQNAgent(env.observation_space.shape, env.action_space.n)

# Training parameters
episodes = 10
max_steps = 100
batch_size = 32

# Training loop
for ep in range(episodes):
    state, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < max_steps:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.remember(state, action, reward, next_state, done)

        if steps % 5 == 0:
            agent.train(batch_size)

        state = next_state
        total_reward += reward
        steps += 1

    print(f"Episode {ep + 1} finished in {steps} steps with total reward {total_reward:.2f}")