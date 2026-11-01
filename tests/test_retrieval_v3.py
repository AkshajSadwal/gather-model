"""Torch checks for the v3 retrieval changes (they run on your Mac; skipped where torch is missing)."""
from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from common.pools import reference_pool_softmax_loss


def small_metadata() -> dict:
    sizes = [max(8, 4 + (i % 4)) for i in range(36)]
    sizes[1] = 30                                   # video-ID table
    return {"dataset": "unit-test", "categorical_sizes": sizes}


@unittest.skipIf(torch is None, "torch is not installed")
class GatedVideoIdTests(unittest.TestCase):
    def setUp(self):
        from model1_retrieval.model import TwoTowerRetrieval
        torch.manual_seed(0)
        self.model = TwoTowerRetrieval(small_metadata())
        self.model.eval()

    def _video(self, ids):
        n = len(ids)
        cat = torch.ones((n, len(self.model.VIDEO_CATALOG_COLUMNS)), dtype=torch.long)
        cat[:, 0] = torch.tensor(ids)
        num = torch.zeros((n, len(self.model.VIDEO_CATALOG_NUMERIC_COLUMNS)))
        return cat, num

    def test_default_behaviour_keeps_every_id(self):
        self.assertTrue(bool(self.model.video_tower.id_keep.all()))
        with torch.no_grad():
            out = self.model.video_tower(*self._video([0, 5, 7]))
        self.assertFalse(torch.allclose(out[1], out[2]))
        self.assertFalse(torch.allclose(out[0], out[1]))

    def test_gated_ids_look_exactly_like_unknown_ids(self):
        keep = np.ones(30, dtype=bool)
        keep[5] = False
        self.model.video_tower.set_id_keep(keep)
        with torch.no_grad():
            out = self.model.video_tower(*self._video([0, 5, 7]))
        self.assertTrue(torch.allclose(out[0], out[1], atol=1e-6))      # gated == unknown
        self.assertFalse(torch.allclose(out[0], out[2], atol=1e-6))     # kept ID still differs
        self.assertFalse(bool(self.model.video_tower.id_keep[0]))       # OOV row is never "kept"

    def test_id_dropout_only_acts_in_training_mode(self):
        tower = self.model.video_tower
        for module in tower.network:                    # remove the MLP's own randomness for a clean comparison
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        cat, num = self._video([7] * 6)
        unknown_cat, unknown_num = self._video([0] * 6)
        with torch.no_grad():
            tower.id_dropout = 0.999999
            tower.eval()
            eval_out = tower(cat, num)                  # eval: ID kept, no dropout
            self.assertFalse(torch.allclose(eval_out, tower(unknown_cat, unknown_num), atol=1e-6))
            tower.train()
            train_out = tower(cat, num)                 # train: ID hidden for (almost) every row
            self.assertTrue(torch.allclose(train_out, tower(unknown_cat, unknown_num), atol=1e-6))
            tower.id_dropout = 0.0
            self.assertTrue(torch.allclose(tower(cat, num), eval_out, atol=1e-6))

    def test_checkpoint_roundtrip_and_legacy_checkpoints(self):
        from model1_retrieval.model import TwoTowerRetrieval
        keep = np.ones(30, dtype=bool)
        keep[[3, 9]] = False
        self.model.video_tower.set_id_keep(keep)
        state = self.model.state_dict()
        self.assertIn("video_tower.id_keep", state)
        fresh = TwoTowerRetrieval(small_metadata())
        fresh.load_state_dict(state)
        self.assertTrue(torch.equal(fresh.video_tower.id_keep, self.model.video_tower.id_keep))
        legacy = {k: v for k, v in state.items() if k != "video_tower.id_keep"}
        old_style = TwoTowerRetrieval(small_metadata())
        old_style.load_state_dict(legacy)                                # must not raise
        self.assertTrue(bool(old_style.video_tower.id_keep[1:].all()))

    def test_bad_keep_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.video_tower.set_id_keep(np.ones(5, dtype=bool))
        with self.assertRaises(ValueError):
            self.model.user_tower.set_id_keep(np.ones(5, dtype=bool))


@unittest.skipIf(torch is None, "torch is not installed")
class PoolSoftmaxLossTests(unittest.TestCase):
    def _case(self, seed=0, b=9, c=14, d=6):
        rng = np.random.default_rng(seed)
        u = rng.normal(size=(b, d)).astype(np.float32); v = rng.normal(size=(c, d)).astype(np.float32)
        u /= np.linalg.norm(u, axis=1, keepdims=True); v /= np.linalg.norm(v, axis=1, keepdims=True)
        own = rng.integers(0, c, b)
        exclude = rng.random((b, c)) < 0.3
        exclude[np.arange(b), own] = False
        logq = rng.normal(size=c).astype(np.float32)
        w = (rng.random(b) + 0.5).astype(np.float32)
        return u, v, own, exclude, logq, w

    def test_matches_numpy_reference(self):
        from model1_retrieval.losses import pool_softmax_loss
        u, v, own, exclude, logq, w = self._case()
        loss, per_row = pool_softmax_loss(torch.from_numpy(u), torch.from_numpy(v), torch.from_numpy(own),
                                          torch.from_numpy(exclude), torch.from_numpy(logq), 0.07, torch.from_numpy(w))
        ref_loss, ref_rows = reference_pool_softmax_loss(u, v, own, exclude, logq, 0.07, w)
        np.testing.assert_allclose(per_row.numpy(), ref_rows, rtol=1e-4, atol=1e-4)
        self.assertAlmostEqual(float(loss), ref_loss, places=4)

    def test_excluded_candidates_do_not_change_the_loss_when_removed(self):
        from model1_retrieval.losses import pool_softmax_loss
        u, v, own, exclude, logq, w = self._case(seed=2)
        base = pool_softmax_loss(torch.from_numpy(u), torch.from_numpy(v), torch.from_numpy(own),
                                 torch.from_numpy(exclude), torch.from_numpy(logq), 0.1, torch.from_numpy(w))[1]
        # make a candidate that is excluded for every row wildly attractive: nothing may change
        exclude2 = exclude.copy(); exclude2[:, 0] = True; exclude2[np.arange(len(own)), own] = False
        v2 = v.copy(); v2[0] = u.mean(axis=0)
        a = pool_softmax_loss(torch.from_numpy(u), torch.from_numpy(v), torch.from_numpy(own),
                              torch.from_numpy(exclude2), torch.from_numpy(logq), 0.1, torch.from_numpy(w))[1]
        b = pool_softmax_loss(torch.from_numpy(u), torch.from_numpy(v2), torch.from_numpy(own),
                              torch.from_numpy(exclude2), torch.from_numpy(logq), 0.1, torch.from_numpy(w))[1]
        rows = ~np.isin(own, [0])        # rows whose own positive is not column 0
        np.testing.assert_allclose(a.numpy()[rows], b.numpy()[rows], rtol=1e-5, atol=1e-5)
        self.assertEqual(base.shape, a.shape)

    def test_gradients_reach_both_towers(self):
        from model1_retrieval.losses import pool_softmax_loss
        from model1_retrieval.model import TwoTowerRetrieval
        torch.manual_seed(1)
        model = TwoTowerRetrieval(small_metadata()); model.train()
        b, c = 6, 10
        u_cat = torch.ones((b, len(model.USER_COLUMNS)), dtype=torch.long)
        u_num = torch.randn((b, len(model.USER_NUMERIC_COLUMNS)))
        v_cat = torch.ones((c, len(model.VIDEO_CATALOG_COLUMNS)), dtype=torch.long)
        v_cat[:, 0] = torch.arange(c) + 2
        v_num = torch.randn((c, len(model.VIDEO_CATALOG_NUMERIC_COLUMNS)))
        own = torch.arange(b)
        exclude = torch.zeros((b, c), dtype=torch.bool)
        loss, _ = pool_softmax_loss(model.user_tower(u_cat, u_num), model.video_tower(v_cat, v_num), own, exclude,
                                    torch.zeros(c), 0.07, torch.ones(b))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for tower in (model.user_tower, model.video_tower):
            grads = [p.grad for p in tower.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)


if __name__ == "__main__":
    unittest.main()
