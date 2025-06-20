import unittest
from src.meta_rl_refactor import MetaRLRefactorAgent, RefactorEnv


class TestMetaRLRefactor(unittest.TestCase):
    def test_q_learning_update(self):
        env = RefactorEnv(['mod'])
        agent = MetaRLRefactorAgent(['mod'], epsilon=0.0)
        action = agent.choose_action('mod')
        reward = env.step('mod', action)
        agent.update('mod', action, reward)
        self.assertNotEqual(agent.q_values['mod'][action], 0.0)


if __name__ == '__main__':
    unittest.main()
