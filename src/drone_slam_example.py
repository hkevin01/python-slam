import cv2
import numpy as np

# NOTE: Requires OpenCV (cv2). Install with: pip install opencv-python

# Initialize feature detector
orb = cv2.ORB_create()

# Load two consecutive frames (replace with actual drone images)
img1 = cv2.imread('frame1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('frame2.png', cv2.IMREAD_GRAYSCALE)

assert img1 is not None and img2 is not None, \
    "Images not found. Please provide 'frame1.png' and 'frame2.png'."

# Detect and compute features
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints
pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

# Camera intrinsics (replace with your drone's calibration)
fx, fy, cx, cy = 700, 700, 320, 240
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Estimate Essential Matrix
E, mask = cv2.findEssentialMat(
    pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Recover relative pose
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

# Triangulate points
proj1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
proj2 = K @ np.hstack((R, t))
pts_3d_hom = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
pts_3d = pts_3d_hom[:3] / pts_3d_hom[3]

print("Estimated 3D points:", pts_3d.T)
