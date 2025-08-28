import gymnasium as gym
import numpy as np

class RubiksCubeEnv(gym.Env):
    def __init__(self):
        super(RubiksCubeEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(54,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(12)  # 6 faces × 2 directions
        self.state = self._get_solved_state()

        # Define face indices
        self.face_indices = {
            'U': list(range(0, 9)),
            'R': list(range(9, 18)),
            'F': list(range(18, 27)),
            'D': list(range(27, 36)),
            'L': list(range(36, 45)),
            'B': list(range(45, 54))
        }

    def _get_solved_state(self):
        return np.array([i // 9 for i in range(54)], dtype=np.int32)

    def reset(self, seed=None, options=None):
        self.state = self._get_solved_state()

        # Scramble the cube with random moves
        for _ in range(10):  # Apply 10 random actions
            action = self.action_space.sample()
            self._apply_action(action)

        return self.state.copy(), {}

    def step(self, action):
        self._apply_action(action)
        reward = 1.0 if self._is_solved() else -0.1
        terminated = self._is_solved()
        truncated = False
        return self.state.copy(), reward, terminated, truncated, {}

    def _is_solved(self):
        return all(len(set(self.state[i:i+9])) == 1 for i in range(0, 54, 9))

    def _rotate_face(self, face, clockwise=True):
        if clockwise:
            return [face[i] for i in [6, 3, 0, 7, 4, 1, 8, 5, 2]]
        else:
            return [face[i] for i in [2, 5, 8, 1, 4, 7, 0, 3, 6]]

    def _apply_action(self, action):
        cube = self.state.copy()
        f = self.face_indices

        if action == 0:  # F (Front face clockwise)
            face = f['F']
            cube[face] = self._rotate_face(cube[face], clockwise=True)

            u, r, d, l = f['U'], f['R'], f['D'], f['L']
            temp = [cube[i] for i in [u[6], u[7], u[8]]]

            cube[u[6]], cube[u[7]], cube[u[8]] = cube[l[8]], cube[l[5]], cube[l[2]]
            cube[l[8]], cube[l[5]], cube[l[2]] = cube[d[2]], cube[d[1]], cube[d[0]]
            cube[d[2]], cube[d[1]], cube[d[0]] = cube[r[0]], cube[r[3]], cube[r[6]]
            cube[r[0]], cube[r[3]], cube[r[6]] = temp[0], temp[1], temp[2]

        # TODO: Add other 11 moves (F', R, R', U, U', D, D', L, L', B, B')

        self.state = cube