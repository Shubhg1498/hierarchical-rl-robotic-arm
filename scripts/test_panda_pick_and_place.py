import time
import gymnasium as gym
import panda_gym


def main():
    env = gym.make("PandaPickAndPlaceDense-v3", render_mode="human")
    obs, info = env.reset()

    print("Environment loaded.")
    print("Observation keys:", obs.keys())
    print("Action space:", env.action_space)

    for _ in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(1 / 60)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()