import unittest

import torch

from model1_retrieval.train import combine_hybrid_losses


class HybridLossTests(unittest.TestCase):
    def test_blends_pairwise_and_inbatch_losses(self):
        pairwise = torch.tensor([0.2, 0.4], requires_grad=True)
        inbatch = torch.tensor([1.0, 2.0], requires_grad=True)
        combined = combine_hybrid_losses(pairwise, inbatch, 0.05)
        expected = 0.95 * pairwise.detach() + 0.05 * inbatch.detach()
        self.assertTrue(torch.allclose(combined.detach(), expected))
        combined.sum().backward()
        self.assertTrue(torch.allclose(pairwise.grad, torch.tensor([0.95, 0.95])))
        self.assertTrue(torch.allclose(inbatch.grad, torch.tensor([0.05, 0.05])))

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            combine_hybrid_losses(torch.ones(2), torch.ones(3), 0.05)

    def test_rejects_invalid_weight(self):
        with self.assertRaises(ValueError):
            combine_hybrid_losses(torch.ones(2), torch.ones(2), 1.0)


if __name__ == "__main__":
    unittest.main()
