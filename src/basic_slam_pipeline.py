"""
Basic SLAM pipeline using ORB feature extraction and Essential Matrix pose estimation.
"""
import cv2
import numpy as np


class BasicSLAMPipeline:
    def __init__(self, camera_matrix):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.K = camera_matrix

    def process_frames(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise FileNotFoundError("Images not found.")
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
        return R, t, matches
