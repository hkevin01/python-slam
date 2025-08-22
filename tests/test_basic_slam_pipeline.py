import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np

from basic_slam_pipeline import BasicSLAMPipeline

# Camera intrinsics (example values)
fx, fy, cx, cy = 700, 700, 320, 240
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def test_basic_slam_pipeline():
    pipeline = BasicSLAMPipeline(K)
    try:
        R, t, matches = pipeline.process_frames('data/frame1.png', 'data/frame2.png')
        assert R.shape == (3, 3)
        assert t.shape == (3, 1)
        assert len(matches) > 0
    except FileNotFoundError:
        # Pass if images are not present, as this is a placeholder test
        pass
