"""Hardware equivalence tests. Run through scripts/validate_metal.py."""
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from adm.config import Config
from adm.composite import compose
from adm.geometry import LocalEarth, intrinsics
from adm.lighting import _choose_cpu, linear_to_srgb, normals_from_height, visibility_from_height
from adm.metadata import FrameMetadata
from adm.pipeline import run
from adm.vocabulary import match_corners


@unittest.skipUnless(os.environ.get('ADM_TEST_METAL') == '1', 'requires an explicitly enabled Metal device')
class MetalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adm.metal import get_backend
        cls.gpu = get_backend()
        cv2.setNumThreads(6)

    def setUp(self):
        self.rng = np.random.default_rng(45)

    def test_shared_view_owns_allocation(self):
        a = np.arange(27, dtype=np.float32).reshape(3, 9)
        view = self.gpu.upload(a).numpy(False)
        gc.collect()
        np.testing.assert_array_equal(view, a)

    def test_dropped_buffers_return_device_memory(self):
        # Regression: C entry points once leaked an autoreleased retain per buffer,
        # so every dropped buffer stayed wired until the machine panicked.
        self.gpu.sync(); gc.collect(); baseline = self.gpu.allocated_bytes()
        for _ in range(64):
            values = self.gpu.upload(np.ones((1024, 1024), np.float32)); norm = self.gpu.buffer(1024)
            self.gpu.dispatch('descriptor_norm', [values, norm], 1024, [1024, 1024]); del values, norm
        self.gpu.sync(); gc.collect()
        self.assertLess(self.gpu.allocated_bytes() - baseline, 16 * 1024**2)
        limit = self.gpu.memory_limit_bytes
        try:
            self.gpu.memory_limit_bytes = self.gpu.allocated_bytes() + 1024
            with self.assertRaisesRegex(MemoryError, 'device memory limit'): self.gpu.buffer(1 << 20)
        finally: self.gpu.memory_limit_bytes = limit

    def test_invalid_gpu_inputs_fail_before_dispatch(self):
        with self.assertRaises(ValueError): self.gpu.descriptors(np.zeros((0, 128)))
        with self.assertRaises(ValueError): self.gpu.descriptors(np.array([[np.nan]], np.float32))
        a=self.gpu.upload(np.zeros((7, 9, 4), np.float32))
        with self.assertRaises(ValueError): self.gpu.gaussian(a, 7, a)
        with self.assertRaises(ValueError): self.gpu.gaussian(a, -1)
        b=self.gpu.descriptors(np.ones((2, 16), np.float32))
        c=self.gpu.descriptors(np.ones((2, 32), np.float32))
        with self.assertRaises(ValueError): self.gpu.match_descriptors(b, c)
        self.gpu.sync()

    def test_mps_gaussian_channels_borders_and_tile_seams(self):
        for h, w, sigma in [(1, 19, .8), (23, 1, 7), (267, 2071, 7), (41, 67, .8)]:
            with self.subTest(shape=(h, w), sigma=sigma):
                a = self.rng.normal(size=(h, w, 4)).astype(np.float32)
                expected = cv2.GaussianBlur(a, (0, 0), sigma)
                actual = self.gpu.gaussian(self.gpu.upload(a), sigma).numpy()
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)

    def test_mps_exact_two_nearest_with_ties_and_batch_tail(self):
        a = self.rng.integers(0, 150, (537, 128)).astype(np.float32)
        b = self.rng.integers(0, 150, (173, 128)).astype(np.float32)
        b[0] = a[0]; b[1] = a[0]; b[-1] = a[-1]
        distance = np.sum((a[:, None] - b[None])**2, axis=2)
        nearest = np.argsort(distance, axis=1, kind='stable')[:, :2]
        actual = self.gpu.match_descriptors(self.gpu.descriptors(a), self.gpu.descriptors(b))
        np.testing.assert_array_equal(actual[:, :2], nearest)
        np.testing.assert_allclose(actual[:, 2:], np.take_along_axis(distance, nearest, axis=1), atol=.1)
        self.assertEqual(self.gpu.match_descriptors(self.gpu.descriptors(a), self.gpu.descriptors(b[:1])).shape, (0, 4))

    def test_tile_padding_channel_order_and_strides(self):
        a = self.rng.integers(0, 256, (47, 59, 3), dtype=np.uint8)[:, ::-1]
        for s in (1, 21, 50):
            rows, cols = (47+s-1)//s, (59+s-1)//s
            expected = np.pad(a[:, :, ::-1], ((0, rows*s-47), (0, cols*s-59), (0, 0)))
            expected = expected.reshape(rows, s, cols, s, 3).transpose(0, 2, 1, 3, 4).reshape(-1, s, s, 3)
            np.testing.assert_array_equal(self.gpu.tiles(a, s), expected)

    def test_warp_rgb_and_generated_edge_confidence(self):
        a = self.rng.integers(0, 256, (117, 173, 3), dtype=np.uint8)
        yy, xx = np.mgrid[:117, :173]
        distance = np.minimum.reduce([xx/(173/2), (172-xx)/(173/2), yy/(117/2), (116-yy)/(117/2)]).astype(np.float32)
        edge = np.clip(distance*8, 0, 1); edge[distance < .02] = 0
        for H in (np.eye(3), np.array([[.93, .04, 12.], [-.07, 1.02, 7.], [.0001, -.0002, 1.]])):
            rgb, weight = self.gpu.warp(self.gpu.upload(a), H, (129, 191))
            expected = cv2.warpPerspective(a, H, (191, 129))
            error = np.abs(rgb.numpy().astype(float)-expected)
            self.assertLess(error.mean(), .02)
            self.assertLessEqual(error.max(), 8)
            np.testing.assert_allclose(weight.numpy(), cv2.warpPerspective(edge, H, (191, 129)), atol=.006)

    def test_corner_acceptance_translation_and_rejection(self):
        a = self.rng.integers(40, 180, (200, 240, 3), dtype=np.uint8)
        b = np.zeros((250, 300, 3), np.uint8); b[17:217, 23:263] = a+15
        H = np.array([[1, 0, 23], [0, 1, 17], [0, 0, 1.]])
        cpu = match_corners(a, b, H, Config())
        metal = match_corners(a, b, H, Config(), gpu=self.gpu)
        self.assertEqual(sum(r['accepted'] for r in metal), 4)
        for c, m in zip(cpu, metal):
            self.assertEqual(c['target'], m['target']); self.assertAlmostEqual(c['ncc'], m['ncc'], places=5)
            self.assertLess(m['delta_rmse'], 1e-6)
        bad = match_corners(a, b, H, Config(), np.eye(3), np.eye(3), gpu=self.gpu)
        self.assertFalse(any(r['accepted'] for r in bad))

    def test_terrain_normals_visibility_and_tiled_lighting(self):
        h, w = 269, 93
        yy, xx = np.mgrid[:h, :w]
        height = (.02*xx + .03*yy + 2*np.sin(xx/13)).astype(np.float32)
        height[:, 42:45] += 7
        rgba = self.rng.integers(30, 230, (h, w, 4), dtype=np.uint8); rgba[:, :, 3] = 255
        rgba[35:45, 13:19, 3] = 0
        cfg = Config(lighting_candidates=3, shadow_steps=29, shadow_max_distance_m=35)
        normal = self.gpu.buffer((h, w, 4))
        self.gpu.dispatch('terrain_normals', [self.gpu.upload(height), normal], h*w, [w, h, 0, h], [.7])
        np.testing.assert_allclose(normal.numpy()[:, :, :3], normals_from_height(height, .7), atol=2e-6)
        expected = _choose_cpu(rgba, cfg, height, .7)
        actual, encoded = self.gpu.lighting(rgba, cfg, height, .7)
        error = np.abs(actual-expected)
        self.assertLess(float(error.mean()), 2e-5)
        self.assertLess(float(np.percentile(error, 99.9)), 2e-5)
        self.assertTrue(np.all(actual[rgba[:, :, 3] == 0] == 0))
        expected_png = np.round(np.clip(linear_to_srgb(expected), 0, 1)[:, :, ::-1]*255).astype(np.uint8)
        self.assertLess(np.abs(encoded[:, :, :3].astype(float)-expected_png).mean(), .02)

    def test_flat_lighting_exact_export_and_repeatability(self):
        rgba = self.rng.integers(0, 256, (79, 113, 4), dtype=np.uint8); rgba[:, :, 3] = 255
        rgba[:4, :, 3] = 0
        for candidates in (1, 5):
            cfg = Config(lighting_candidates=candidates)
            expected = _choose_cpu(rgba, cfg)
            actual, encoded = self.gpu.lighting(rgba, cfg)
            np.testing.assert_allclose(actual, expected, atol=4e-7)
            expected_png = np.round(np.clip(linear_to_srgb(expected), 0, 1)[:, :, ::-1]*255).astype(np.uint8)
            # Float pow can straddle a half-integer during 8-bit quantization.
            error=np.abs(encoded[:, :, :3].astype(float)-expected_png)
            self.assertLessEqual(error.max(), 1)
            self.assertLess(error.mean(), .001)

    def test_composite_seams_coverage_source_ids_and_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cpu = root/'cpu'; metal = root/'metal'; cpu.mkdir(); metal.mkdir()
            base = self.rng.integers(40, 160, (187, 323, 3), dtype=np.uint8)
            names = ['a.png', 'b.png']; cv2.imwrite(str(root/names[0]), base[:, :251]); cv2.imwrite(str(root/names[1]), base[:, 72:]+12)
            meta = [FrameMetadata(n, 251, 187) for n in names]
            hs = [np.eye(3), np.array([[1, 0, 72], [0, 1, 0], [0, 0, 1.]])]
            cfg = Config(max_output_dimension=500)
            a, _, _ = compose(root, names, meta, hs, cfg, cpu)
            b, _, _ = compose(root, names, meta, hs, cfg, metal, gpu=self.gpu)
            np.testing.assert_array_equal(a[:, :, 3], b[:, :, 3])
            error = np.abs(a[:, :, :3].astype(float)-b[:, :, :3])
            self.assertLess(error.mean(), .1)
            ca = cv2.imread(str(cpu/'coverage.png'), -1); cb = cv2.imread(str(metal/'coverage.png'), -1)
            np.testing.assert_array_equal(ca, cb)
            ia = cv2.imread(str(cpu/'source_ids.png'), -1); ib = cv2.imread(str(metal/'source_ids.png'), -1)
            self.assertLess(np.mean(ia != ib), .0001)
            repeated, _, _ = compose(root, names, meta, hs, cfg, metal, gpu=self.gpu)
            np.testing.assert_array_equal(b, repeated)
            cfg.metal_recompute_warps=True
            lowdisk=root/'lowdisk'; lowdisk.mkdir()
            recomputed, _, _=compose(root, names, meta, hs, cfg, lowdisk, gpu=self.gpu)
            np.testing.assert_array_equal(b, recomputed)
            self.assertFalse((lowdisk/'warps').exists())

    def test_geographic_end_to_end_and_resume(self):
        from test_pipeline import write_exif
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'source'; source.mkdir(); out = root/'result'
            base = cv2.GaussianBlur(self.rng.integers(30, 210, (500, 1100, 3), dtype=np.uint8), (3, 3), .6)
            cfg = Config(skip_first=0, registration_width=800, max_output_dimension=700, feature_count=3000)
            earth = LocalEarth(35, -97); dummy = FrameMetadata('a', 800, 500, 35, -97, 0)
            gsd = cfg.altitude_agl_m/intrinsics(dummy, cfg)[0, 0]
            for i in range(3):
                lat, lon = earth.latlon(np.array([i*100*gsd, 0.]))
                write_exif(source/f'{i}.jpg', base[:, i*100:i*100+800], lat, lon, 0)
            config = root/'config.json'; config.write_text(json.dumps(cfg.to_dict()))
            args = SimpleNamespace(source=source, output=out, config=config, altitude=None, max_dim=None,
                                   strict_corners=False, exclusions=None, metadata=None, allow_local=False,
                                   resume=False, heightmap=None, heightmap_spacing=None, backend='metal')
            run(args)
            report = json.loads((out/'report.json').read_text())
            self.assertTrue(report['georeferenced']); self.assertEqual(report['registered_pairs'], 3)
            self.assertGreater(report['acceleration']['mps_operations'], 0)
            self.assertLess(json.loads((out/'registration_report.json').read_text())['p95_registration_error_px'], .25)
            first = cv2.imread(str(out/'map_golden.png'), -1)
            cache_time = (out/'registration.npz').stat().st_mtime_ns
            args.resume = True; run(args)
            self.assertEqual(cache_time, (out/'registration.npz').stat().st_mtime_ns)
            np.testing.assert_array_equal(first, cv2.imread(str(out/'map_golden.png'), -1))
