import unittest
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import run_tiny_sanity

class TestTinySanityConfig(unittest.TestCase):
    def test_tiny_sanity_configuration(self):
        """
        Regression test ensuring Tiny Sanity parameters match the V2 specification:
        - DATASET_SIZE == 200
        - EPOCHS == 100
        - BATCH_SIZE == 4
        - EXPECTED_OPTIMIZER_STEPS == 5000
        """
        self.assertEqual(run_tiny_sanity.DATASET_SIZE, 200)
        self.assertEqual(run_tiny_sanity.EPOCHS, 100)
        self.assertEqual(run_tiny_sanity.BATCH_SIZE, 4)
        self.assertEqual(run_tiny_sanity.MAX_SEQ_LEN, 64)
        self.assertEqual(run_tiny_sanity.OBJECTIVE_MODE, "direct")

        batches_per_epoch = math.ceil(run_tiny_sanity.DATASET_SIZE / run_tiny_sanity.BATCH_SIZE)
        self.assertEqual(batches_per_epoch, 50)

        expected_optimizer_steps = batches_per_epoch * run_tiny_sanity.EPOCHS
        self.assertEqual(expected_optimizer_steps, 5000)

if __name__ == "__main__":
    unittest.main()
