import numpy as np
import gymnasium as gym

class EgocentricRaycastWrapper(gym.ObservationWrapper):
    """
    EM-NAV Custom Wrapper: Overrides standard MiniGrid observations to enforce
    a strict, continuous 5-ray egocentric distance sensory stream.
    """
    def __init__(self, env, max_range=8.0):
        super().__init__(env)
        self.max_range = max_range
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.relative_angles = [-90, -45, 0, 45, 90]

    def observation(self, obs):
        grid = self.env.unwrapped.grid
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir  # 0:R, 1:D, 2:L, 3:U
       
        base_angle = agent_dir * 90
        distances = []
       
        for rel_angle in self.relative_angles:
            target_angle = (base_angle + rel_angle) % 360
            dist = self._ray_march(agent_pos, target_angle, grid)
            distances.append(dist / self.max_range)
           
        return np.array(distances, dtype=np.float32)

    def _ray_march(self, start_pos, angle_deg, grid):
        rad = np.radians(angle_deg)
        dx, dy = np.cos(rad), np.sin(rad)
       
        for step in range(1, int(self.max_range) + 1):
            curr_x = int(round(start_pos[0] + dx * step))
            curr_y = int(round(start_pos[1] + dy * step))
           
            if not (0 <= curr_x < grid.width and 0 <= curr_y < grid.height):
                return float(step)
               
            cell = grid.get(curr_x, curr_y)
            if cell is not None and cell.type in ['wall', 'door']:
                return float(step)
               
        return self.max_range