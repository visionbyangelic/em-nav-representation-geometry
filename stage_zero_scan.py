import numpy as np
from minigrid.envs import EmptyEnv
from minigrid.core.world_object import Wall
from wrappers.raycast import EgocentricRaycastWrapper

def execute_stage_zero_scan(maze_size=12):
    """
    Runs an absolute information-theoretic scan of the workspace to compute
    the structural baseline perceptual confusion index before training.
    """
    # 1. Initialize custom 12x12 grid sandbox directly
    base_env = EmptyEnv(size=maze_size, render_mode=None)
    base_env.reset()
    grid = base_env.unwrapped.grid
   
    # 2. Inject your internal partition wall to build structural obstacles
    for y in range(2, maze_size - 2):
        grid.set(maze_size // 2, y, Wall())
       
    # 3. Apply your modular custom sensory wrapper
    wrapper = EgocentricRaycastWrapper(base_env)
   
    observation_registry = {}
    total_valid_states = 0
   
    # 4. Sweep every tile and direction across the matrix
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type in ['wall', 'door']:
                continue
               
            for heading in range(4):
                base_env.unwrapped.agent_pos = (x, y)
                base_env.unwrapped.agent_dir = heading
               
                obs = wrapper.observation(None)
                obs_key = tuple(np.round(obs, decimals=4))
               
                state_data = {'pos': (x, y), 'dir': heading}
                if obs_key not in observation_registry:
                    observation_registry[obs_key] = []
                observation_registry[obs_key].append(state_data)
                total_valid_states += 1
               
    aliased_groups = {k: v for k, v in observation_registry.items() if len(v) > 1}
    aliased_states_count = sum([len(v) for v in aliased_groups.values()])
   
    aliasing_density = (aliased_states_count / total_valid_states) * 100
   
    asi_scores = []
    for states in aliased_groups.values():
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                p1, p2 = states[i]['pos'], states[j]['pos']
                geodesic_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                asi_scores.append(geodesic_distance)
               
    print(f"====================================================")
    print(f"📋 PRE-REGISTRATION STATUS REPORT: STAGE 0 COMPLETED")
    print(f"====================================================")
    print(f"Total Trajectory State Profiles Mapped : {total_valid_states}")
    print(f"Environmental Perceptual Aliasing Density : {aliasing_density:.2f}%")
    if len(asi_scores) > 0:
        print(f"Mean Alias Severity Index (ASI) Value : {np.mean(asi_scores):.2f} cells")
        print(f"Maximum Alias Severity Index (ASI) Range : {np.max(asi_scores):.2f} cells")
    print(f"====================================================")

if __name__ == "__main__":
    execute_stage_zero_scan(maze_size=12)