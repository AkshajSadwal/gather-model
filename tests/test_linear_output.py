"""Linear-output option of the retrieval towers. Needs torch (run on the Mac: python -m unittest tests.test_linear_output)."""
import unittest

import torch

from model1_retrieval.model import TwoTowerRetrieval

META = {"dataset": "unit-test", "categorical_sizes": [max(8, 4 + (i % 4)) for i in range(36)]}


def _inputs(model, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    cat = torch.randint(1, 4, (n, len(model.VIDEO_COLUMNS)), generator=g)
    num = torch.randn(n, len(model.VIDEO_NUMERIC_COLUMNS), generator=g)
    return cat, num


class LinearOutputTests(unittest.TestCase):
    def test_legacy_towers_are_non_negative_and_linear_towers_are_not(self):
        torch.manual_seed(0)
        legacy, linear = TwoTowerRetrieval(META), TwoTowerRetrieval(META, linear_output=True)
        legacy.eval(); linear.eval()
        cat, num = _inputs(legacy)
        user_cat = torch.randint(1, 4, (64, len(legacy.USER_COLUMNS)))
        user_num = torch.randn(64, len(legacy.USER_NUMERIC_COLUMNS))
        with torch.no_grad():
            self.assertGreaterEqual(float(legacy.video_tower(cat, num).min()), 0.0)
            self.assertGreaterEqual(float(legacy.user_tower(user_cat, user_num).min()), 0.0)
            video_vec = linear.video_tower(cat, num)
            user_vec = linear.user_tower(user_cat, user_num)
        self.assertLess(float(video_vec.min()), 0.0)
        self.assertLess(float(user_vec.min()), 0.0)
        self.assertTrue(torch.allclose(video_vec.norm(dim=1), torch.ones(video_vec.shape[0]), atol=1e-5))
        self.assertTrue(torch.allclose(user_vec.norm(dim=1), torch.ones(user_vec.shape[0]), atol=1e-5))

    def test_flag_travels_with_the_state_dict(self):
        source = TwoTowerRetrieval(META, linear_output=True)
        target = TwoTowerRetrieval(META)             # built with the default, as every loader does
        target.load_state_dict(source.state_dict())
        self.assertTrue(target.user_tower.linear_output and target.video_tower.linear_output)
        source.eval(); target.eval()
        cat, num = _inputs(source)
        with torch.no_grad():
            self.assertTrue(torch.allclose(source.video_tower(cat, num), target.video_tower(cat, num), atol=1e-6))

    def test_checkpoints_saved_before_the_flag_still_load_with_relu(self):
        legacy = TwoTowerRetrieval(META)
        state = {k: v for k, v in legacy.state_dict().items() if not k.endswith("linear_output_flag")}
        fresh = TwoTowerRetrieval(META, linear_output=True)   # even if constructed linear, the old file wins
        fresh.load_state_dict(state)
        self.assertFalse(fresh.user_tower.linear_output or fresh.video_tower.linear_output)
        legacy.eval(); fresh.eval()
        cat, num = _inputs(legacy)
        user_cat = torch.randint(1, 4, (64, len(legacy.USER_COLUMNS)))
        user_num = torch.randn(64, len(legacy.USER_NUMERIC_COLUMNS))
        with torch.no_grad():
            self.assertTrue(torch.allclose(legacy.video_tower(cat, num), fresh.video_tower(cat, num), atol=1e-6))
            self.assertTrue(torch.allclose(
                legacy.user_tower(user_cat, user_num),
                fresh.user_tower(user_cat, user_num),
                atol=1e-6,
            ))


if __name__ == "__main__":
    unittest.main()
