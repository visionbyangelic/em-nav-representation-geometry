"""
========================================================================================================
FILE: wrappers/raycast.py
MODULE: Egocentric Raycasting Sensory Stream Wrapper
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

RESEARCH & SCIENTIFIC PURPOSE:
  This module implements the core sensory input pipeline for the EM-NAV study.
  It strips away all global coordinates (x, y), compass directions, global orientation maps,
  and goal-pointing vectors from the Gymnasium / MiniGrid environment.
  
  Instead, the agent experiences the maze exclusively through a continuous 5-ray egocentric
  distance array:
      x_t = [d_left, d_diag_left, d_front, d_diag_right, d_right] ∈ [0.0, 1.0]^5
      
  Rays measure continuous normalized distance up to a max range of 8.0 grid units at relative
  angles [-90°, -45°, 0°, +45°, +90°].

PERCEPTUAL ALIASING ENFORCEMENT:
  Because physically distant corridors and corners yield identical 5-ray sensor distance vectors,
  the environment exhibits severe perceptual aliasing (81.52% aliasing density). This forces
  the agent to form internal spatial representations to resolve location ambiguity.

INPUT / OUTPUT SPECIFICATIONS:
  - Input: Raw Gymnasium environment step / reset state.
  - Output: 5-element float32 NumPy array x_t ∈ [0.0, 1.0]^5.
========================================================================================================
"""

import numpy as np
import gymnasium as gym


class EgocentricRaycastWrapper(gym.ObservationWrapper):
    """
    Gymnasium Observation Wrapper enforcing a 5-ray continuous egocentric distance stream.
    
    Attributes:
        max_range (float): Maximum raycast sensing distance in grid units (default: 8.0).
        observation_space (gym.spaces.Box): Continuous 5-element box bounded in [0.0, 1.0].
        relative_angles (list[int]): Raycast emission angles [-90°, -45°, 0°, +45°, +90°].
    """
    def __init__(self, env, max_range=8.0):
        super().__init__(env)
        self.max_range = max_range
        # Define 5-element continuous observation space bounded in [0.0, 1.0]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.relative_angles = [-90, -45, 0, 45, 90]

    def observation(self, obs):
        """
        Calculates 5-ray continuous obstacle distance readings relative to agent heading.
        
        Heading Mapping:
            0: Right (0°) | 1: Down (90°) | 2: Left (180°) | 3: Up (270°)
        """
        grid = self.env.unwrapped.grid
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir  # 0:R, 1:D, 2:L, 3:U

        # Compute absolute orientation angle in degrees
        base_angle = agent_dir * 90
        distances = []

        # March rays at relative angles [-90°, -45°, 0°, +45°, +90°]
        for rel_angle in self.relative_angles:
            target_angle = (base_angle + rel_angle) % 360
            dist = self._ray_march(agent_pos, target_angle, grid)
            # Normalize distance to range [0.0, 1.0]
            distances.append(dist / self.max_range)

        return np.array(distances, dtype=np.float32)

    def _ray_march(self, start_pos, angle_deg, grid):
        """
        Marches a ray step-by-step from start_pos until encountering a wall or max_range.
        
        Args:
            start_pos (tuple[int, int]): Agent (x, y) coordinate.
            angle_deg (float): Ray angle in degrees.
            grid (Grid): MiniGrid environment grid.
            
        Returns:
            float: Distance to nearest obstacle in grid units (up to max_range).
        """
        rad = np.radians(angle_deg)
        dx, dy = np.cos(rad), np.sin(rad)

        for step in range(1, int(self.max_range) + 1):
            curr_x = int(round(start_pos[0] + dx * step))
            curr_y = int(round(start_pos[1] + dy * step))

            # Boundary check: Ray hits environment perimeter
            if not (0 <= curr_x < grid.width and 0 <= curr_y < grid.height):
                return float(step)

            # Cell check: Ray hits interior wall or door
            cell = grid.get(curr_x, curr_y)
            if cell is not None and cell.type in ['wall', 'door']:
                return float(step)

        return self.max_range