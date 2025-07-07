import importlib.machinery
import importlib.util
import sys
import types
import unittest
import torch

pkg = types.ModuleType('asi')
sys.modules['asi'] = pkg


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'asi'
    loader.exec_module(mod)
    sys.modules[name] = mod
    setattr(pkg, name.split('.')[-1], mod)
    return mod


got = load('asi.graph_of_thought', 'src/graph_of_thought.py')
mem_mod = load('asi.gnn_memory', 'src/gnn_memory.py')

GraphOfThought = got.GraphOfThought
GNNMemory = mem_mod.GNNMemory


class TestGNNMemory(unittest.TestCase):
    def test_training_closeness(self):
        torch.manual_seed(0)
        g = GraphOfThought()
        a = g.add_step('a')
        b = g.add_step('b')
        c = g.add_step('c')
        d = g.add_step('d')
        g.connect(a, b)
        g.connect(b, c)
        nodes = list(g.nodes.values())
        mem = GNNMemory(nodes, g.edges, dim=8)
        opt = torch.optim.SGD(mem.parameters(), lr=0.1)
        for _ in range(100):
            opt.zero_grad()
            loss = mem.edge_loss()
            loss.backward()
            opt.step()
        emb = mem.encode_nodes().detach()
        idx_a = mem.id_to_idx[a]
        idx_b = mem.id_to_idx[b]
        idx_d = mem.id_to_idx[d]
        sim_ab = torch.cosine_similarity(emb[idx_a], emb[idx_b], dim=0)
        sim_ad = torch.cosine_similarity(emb[idx_a], emb[idx_d], dim=0)
        self.assertGreater(sim_ab, sim_ad)


if __name__ == '__main__':
    unittest.main()
