import unittest
from asi.training_anomaly_detector import TrainingAnomalyDetector

class TestTrainingAnomalyDetector(unittest.TestCase):
    def test_record(self):
        det = TrainingAnomalyDetector(window=3, threshold=1.5)
        self.assertFalse(det.record(1.0))
        det.record(1.0)
        self.assertTrue(det.record(3.0))

if __name__ == '__main__':
    unittest.main()
