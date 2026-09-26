import unittest

import torch
from torch import nn

from src.model.attention import CausalSelfAttention
from src.model.embeddings import GPTEmbedding
from src.model.feedforward import FeedForward
from src.model.gpt import GPT


class DropoutTests(unittest.TestCase):
    def make_model(self, dropout):
        return GPT(32, 8, 16, 4, 2, 32, dropout=dropout)

    def test_training_and_evaluation(self):
        torch.manual_seed(0)
        model = self.make_model(0.5)
        tokens = torch.randint(0, 32, (2, 8))
        model.train()
        first = model(tokens)
        self.assertEqual(first.shape, (2, 8, 32))
        self.assertFalse(torch.equal(first, model(tokens)))
        first.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        model.eval()
        self.assertTrue(torch.equal(model(tokens), model(tokens)))

    def test_zero_dropout_is_deterministic_during_training(self):
        model = self.make_model(0.0).train()
        tokens = torch.randint(0, 32, (2, 8))
        self.assertTrue(torch.equal(model(tokens), model(tokens)))

    def test_rate_reaches_all_layers(self):
        model = self.make_model(0.3)
        layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        self.assertEqual(len(layers), 9)
        self.assertTrue(all(layer.p == 0.3 for layer in layers))

    def test_dropout_is_applied_to_component_outputs(self):
        tokens = torch.zeros((2, 8), dtype=torch.long)
        hidden = torch.randn(2, 8, 16)
        for module, inputs in (
            (GPTEmbedding(32, 16, 8, dropout=1.0), tokens),
            (CausalSelfAttention(16, 4, dropout=1.0), hidden),
            (FeedForward(16, 32, dropout=1.0), hidden),
        ):
            self.assertEqual(torch.count_nonzero(module(inputs)).item(), 0)
            module.eval()
            self.assertGreater(torch.count_nonzero(module(inputs)).item(), 0)


if __name__ == "__main__":
    unittest.main()
