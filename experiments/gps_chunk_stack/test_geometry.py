"""Coordinate orientation and tile-cut parity regressions; no Metal dependency."""
import unittest

import numpy as np

from run import LocalEarth, camera_to_ground, chunk_grid, composite_tile, project


class GeometryTests(unittest.TestCase):
    def test_nadir_center_and_camera_axes(self):
        earth = LocalEarth(36, -95)
        meta = dict(name='fixture', width=101, height=81, latitude=36, longitude=-95,
                    yaw_deg=0, pitch_deg=-90, roll_deg=0, altitude_agl_m=50)
        H = camera_to_ground(meta, earth, 90)
        center, right, top = project(H, [[50, 40], [60, 40], [50, 30]])
        np.testing.assert_allclose(center, 0, atol=1e-12)
        self.assertGreater(right[0], 0)
        self.assertGreater(top[1], 0)
        meta['yaw_deg'] = 90
        H = camera_to_ground(meta, earth, 90)
        right, top = project(H, [[60, 40], [50, 30]])
        self.assertLess(right[1], 0)
        self.assertGreater(top[0], 0)

    def test_coordinate_round_trip(self):
        earth = LocalEarth(36, -95)
        xy = np.array([[-400, 300], [0, 0], [145, -239]])
        ll = earth.latlon(xy)
        np.testing.assert_allclose(earth.xy(ll[:, 0], ll[:, 1]), xy, atol=1e-7)

    def test_horizon_rejected(self):
        meta = dict(name='fixture', width=101, height=81, latitude=36, longitude=-95,
                    yaw_deg=0, pitch_deg=-5, roll_deg=0, altitude_agl_m=50)
        with self.assertRaisesRegex(ValueError, 'horizon'):
            camera_to_ground(meta, LocalEarth(36, -95), 90)

    def test_partial_tiles_cover_source_exactly_once(self):
        rects, centers, corners = chunk_grid(117, 83, 50)
        counts = np.zeros((83, 117), np.uint8)
        for x, y, w, h in rects:
            counts[y:y+h, x:x+w] += 1
        np.testing.assert_array_equal(counts, 1)
        np.testing.assert_allclose(centers, corners.mean(axis=1))

    def test_tiled_warp_equals_uncut_warp(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (83, 117, 3), dtype=np.uint8)
        H = np.array([[1.08, -.23, 40], [.17, 1.13, 10], [.0002, -.0001, 1.]])
        inverse = np.linalg.inv(H)

        def buffers():
            return (np.zeros((140, 190, 3), np.uint8), np.zeros((140, 190), np.float32),
                    np.zeros((140, 190, 3), np.float32), np.zeros((140, 190), np.float32),
                    np.zeros((140, 190), np.uint16), np.full((140, 190), 65535, np.uint16), 0)

        full, tiles = buffers(), buffers()
        composite_tile(image, [0, 0, 117, 83], H, inverse, full)
        for rect in chunk_grid(117, 83, 50)[0]:
            composite_tile(image, rect, H, inverse, tiles)
        for actual, expected in zip(tiles[:6], full[:6]):
            np.testing.assert_array_equal(actual, expected)
        self.assertLessEqual(tiles[4].max(), 1)


if __name__ == '__main__':
    unittest.main()
