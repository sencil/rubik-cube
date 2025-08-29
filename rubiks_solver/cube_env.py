import numpy as np
import gym
from gym import spaces

class RubiksCubeEnv(gym.Env):
    def __init__(self):
        super(RubiksCubeEnv, self).__init__()
        self.action_space = spaces.Discrete(12)  # 6 faces × 2 directions
        self.observation_space = spaces.Box(low=0, high=5, shape=(54,), dtype=np.int32)
        self.state = self._get_solved_state()
        self.face_indices = self._get_face_indices()
        self.phase = 1  # Subproblem phase tracker

    def reset(self):
        self.state = self._scramble_cube()
        self.phase = 1
        return self.state

    def step(self, action):
        self._apply_action(action)
        reward = self._calculate_reward()
        done = self._is_solved()
        return self.state, reward, done, {}

    def _get_solved_state(self):
        return np.array([i // 9 for i in range(54)], dtype=np.int32)

    def _scramble_cube(self, moves=10):
        state = self._get_solved_state()
        for _ in range(moves):
            self._apply_action(np.random.randint(12))
        return state

    def _apply_action(self, action):
        # Placeholder: implement actual cube rotation logic
        pass

    def _is_solved(self):
        return all(len(set(self.state[i:i+9])) == 1 for i in range(0, 54, 9))

    def _get_face_indices(self):
        return {
            'U': list(range(0, 9)),
            'R': list(range(9, 18)),
            'F': list(range(18, 27)),
            'D': list(range(27, 36)),
            'L': list(range(36, 45)),
            'B': list(range(45, 54))
        }

    def _count_uniform_faces(self):
        return sum(len(set(self.state[i:i+9])) == 1 for i in range(0, 54, 9))

    def _has_white_cross(self):
        u = self.face_indices['U']
        center = self.state[u[4]]
        return all(self.state[i] == center for i in [u[1], u[3], u[5], u[7]])

    def _calculate_reward(self):
        reward = 0.0

        # Subgoal: reward for each fully solved face
        reward += 0.1 * self._count_uniform_faces()

        # Pattern: reward for forming white cross
        if self._has_white_cross():
            reward += 0.2

        # Subproblem: phase-based reward shaping
        if self.phase == 1 and self._count_uniform_faces() >= 1:
            reward += 0.3
            self.phase = 2
        elif self.phase == 2 and self._count_uniform_faces() >= 3:
            reward += 0.5
            self.phase = 3
        elif self.phase == 3 and self._is_solved():
            reward += 1.0

        # Small penalty to encourage efficiency
        reward -= 0.05

        return reward
