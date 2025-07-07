import importlib.machinery
import importlib.util
import sys
import unittest
import torch
import numpy as np

loader = importlib.machinery.SourceFileLoader('evds', 'src/event_sensor_dataset.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
evds = importlib.util.module_from_spec(spec)
loader.exec_module(evds)
EventSensorDataset = evds.EventSensorDataset
load_synthetic_events = evds.load_synthetic_events


class TestEventSensorDataset(unittest.TestCase):
    def test_synthetic_events(self):
        ds = load_synthetic_events(num_samples=2, channels=2, length=8)
        self.assertEqual(len(ds), 2)
        sample = ds[0]
        self.assertEqual(sample.shape, (2, 8))
        self.assertTrue(torch.is_tensor(sample))

    def test_load_from_file(self):
        import tempfile
        arr = torch.ones(2, 4).numpy().astype("float32")
        with tempfile.NamedTemporaryFile(suffix=".npy") as f:
            np.save(f.name, arr)
            ds = EventSensorDataset([f.name], normalize=False)
            self.assertEqual(len(ds), 1)
            loaded = ds[0]
            self.assertTrue(torch.allclose(loaded, torch.from_numpy(arr)))


if __name__ == '__main__':
    unittest.main()
