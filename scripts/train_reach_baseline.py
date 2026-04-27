import os
import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy


LOG_DIR = "logs/panda_reach_sac"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/sac_panda_reach"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def plot_rewards():
    x, y = ts2xy(load_results(LOG_DIR), "timesteps")

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("PandaReach-v3 SAC Training Reward")
    plt.grid(True)
    plt.savefig("results/panda_reach_training_curve.png")
    plt.show()


def main():
    env = gym.make("PandaReach-v3")
    env = Monitor(env, LOG_DIR)

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
    )

    model.learn(total_timesteps=50_000)
    model.save(MODEL_PATH)

    env.close()
    plot_rewards()

    print(f"Model saved to: {MODEL_PATH}")
    print("Training curve saved to: results/panda_reach_training_curve.png")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()