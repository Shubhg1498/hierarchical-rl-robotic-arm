# Hierarchical Reinforcement Learning for Robotic Arm Pick-and-Place

## Project Goal

The goal of this project is to develop a hierarchical reinforcement learning system for a simulated robotic arm performing a pick-and-place task.

Instead of training one large policy directly, the task will be decomposed into smaller skills such as reaching, grasping, lifting, moving, and placing. A higher-level policy or selector will later decide which skill to execute depending on the current state.

## Final Demonstration

The final demo should show a simulated robotic arm that can:

1. Move toward an object
2. Align the gripper
3. Grasp the object
4. Lift the object
5. Move it to a target location
6. Place it successfully

## Learning Objectives

- Understand robotic manipulation environments
- Learn continuous-control reinforcement learning
- Train RL agents using Stable-Baselines3
- Use PyBullet / panda-gym simulation
- Design reward functions for robotic skills
- Implement hierarchical reinforcement learning
- Visualize training performance using plots
- Save trained models and demonstrate behavior

## Planned Environment

Initial environment:

- PandaReach-v3

Later environments:

- PandaPickAndPlace-v3
- Custom hierarchical wrapper

## Planned Skills

| Skill | Description |
|---|---|
| ReachSkill | Move end-effector close to the object |
| GraspSkill | Align gripper and close around object |
| LiftSkill | Lift object above the table |
| MoveToTargetSkill | Carry object near target |
| PlaceSkill | Lower and release object |

## Project Phases

### Phase 1: Setup
- Create repository
- Install dependencies
- Run basic panda-gym environment
- Verify rendering

### Phase 2: Baseline RL
- Train a standard RL agent on PandaReach-v3
- Plot reward curves
- Save and evaluate model

### Phase 3: Pick-and-Place Baseline
- Train PPO / SAC on PandaPickAndPlace-v3
- Evaluate success rate
- Record demo video

### Phase 4: Skill Decomposition
- Define individual skill wrappers
- Modify rewards for each skill
- Train low-level policies

### Phase 5: Hierarchical Policy
- Create a high-level selector
- First implement rule-based selector
- Later replace with learned selector

### Phase 6: Final Demo
- Run full pick-and-place sequence
- Save plots, videos, and results
- Write final README and project explanation

## Success Criteria

The project is successful if:

- The arm can complete at least one pick-and-place sequence in simulation
- Training curves are logged and visualized
- The repository contains clear documentation
- The final behavior can be demonstrated through video or live rendering
