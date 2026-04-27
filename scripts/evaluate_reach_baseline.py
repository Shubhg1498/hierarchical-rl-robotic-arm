import time
import numpy as np
import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC


MODEL_PATH = "models/sac_panda_reach"


def distance(obs):
    achieved = obs["achieved_goal"]
    desired = obs["desired_goal"]
    return np.linalg.norm(achieved - desired)


def main():
    env = gym.make("PandaReach-v3", render_mode="human")
    model = SAC.load(MODEL_PATH)

    obs, info = env.reset()

    good_actions = 0
    bad_actions = 0
    successes = 0
    total_steps = 0

    distances = []
    prev_dist = distance(obs)

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        current_dist = distance(obs)
        distances.append(current_dist)

        if current_dist < prev_dist:
            good_actions += 1
            cue = "✅ closer"
        else:
            bad_actions += 1
            cue = "❌ farther"

        total_steps += 1

        if step % 20 == 0:
            print(
                f"Step {step:04d} | "
                f"Distance: {current_dist:.4f} | "
                f"Reward: {reward:.4f} | "
                f"{cue}"
            )

        prev_dist = current_dist
        time.sleep(1 / 60)

        if terminated or truncated:
            if info.get("is_success", False):
                successes += 1
                print("🎯 SUCCESS: target reached")
            else:
                print("Episode ended without success")

            obs, info = env.reset()
            prev_dist = distance(obs)

    env.close()

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Total steps: {total_steps}")
    print(f"Good actions: {good_actions}")
    print(f"Bad actions: {bad_actions}")
    print(f"Good action ratio: {good_actions / total_steps:.2%}")
    print(f"Successes: {successes}")

    plt.figure()
    plt.plot(distances)
    plt.xlabel("Step")
    plt.ylabel("Distance to Target")
    plt.title("End-Effector Distance to Target During Evaluation")
    plt.grid(True)
    plt.savefig("results/reach_distance_evaluation.png")
    plt.show()


if __name__ == "__main__":
    main()