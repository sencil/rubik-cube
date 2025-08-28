## Rubik's Cube Solver with Deep Q-Learning

This project trains a reinforcement learning agent to solve a virtual Rubik’s Cube using Deep Q-Learning (DQN). It leverages the `gymnasium` environment for cube simulation and `TensorFlow` for building and training the neural network.

### Key Components

-`cube_env.py`: Defines the Rubik’s Cube environment using `gymnasium`. It handles cube states, actions (rotations), and reward logic.
-`dqn_agent.py`: Implements the DQN agent. It includes the neural network architecture, experience replay, and action selection strategy.
-`train.py`: The main training loop. It runs episodes where the agent interacts with the environment, learns from experience, and improves its solving strategy.

### Output

During training, the script prints metrics like:

- Episode number: Shows which training cycle the agent is currently running, helping track its overall progress.
- Steps taken: Indicates how many moves the agent made in that episode before reaching a terminal state.
- Reward earned: Reflects how successful the agent’s actions were, with higher rewards meaning better performance.

These help track the agent’s learning progress. You can also modify the code to save model checkpoints or visualize cube-solving performance.