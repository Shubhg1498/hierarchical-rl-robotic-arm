import os
import gymnasium as gym
import panda_gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


LOG_DIR = "logs/panda_pick_place_sac"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/sac_panda_pick_place"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    env = gym.make("PandaPickAndPlaceDense-v3")
    env = Monitor(env, LOG_DIR)

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        train_freq=1,
        gradient_steps=1,
    )

    model.learn(total_timesteps=200_000)
    model.save(MODEL_PATH)

    env.close()
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()