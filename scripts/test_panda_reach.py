import gymnasium as gym
import panda_gym


def main():
    env = gym.make("PandaReach-v3", render_mode="human")

    observation, info = env.reset()

    print("Environment loaded successfully.")
    print("Observation keys:", observation.keys())
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()