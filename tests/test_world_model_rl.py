import unittest
import gym
import torch

from asi.world_model_rl import WorldModel, RandomPolicy, collect_dataset, train_world_model, rollout


class TestWorldModelRL(unittest.TestCase):
    def test_training_and_rollout(self):
        env = gym.make("CartPole-v1")
        policy = RandomPolicy(action_dim=env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n)
        dataset = collect_dataset(env, policy, episodes=1)
        obs_dim = env.observation_space.shape[0]
        action_dim = policy.action_dim
        model = WorldModel(obs_dim, action_dim)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_world_model(dataset, model, opt, epochs=1)
        start = torch.tensor(env.reset()[0], dtype=torch.float32)
        states, rewards = rollout(model, policy, start, horizon=3)
        self.assertEqual(len(states), 4)
        self.assertEqual(len(rewards), 3)


if __name__ == "__main__":
    unittest.main()
