import numpy as np

from pose_estimation import PoseEstimator


def test_pose_estimation():
    estimator = PoseEstimator()
    # Dummy keypoints and camera matrix
    keypoints1 = np.random.rand(10, 2)
    keypoints2 = np.random.rand(10, 2)
    camera_matrix = np.eye(3)
    pose = estimator.estimate(keypoints1, keypoints2, camera_matrix)
    assert pose is None or isinstance(pose, tuple)
