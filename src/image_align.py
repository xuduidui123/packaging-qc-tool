"""
图像预处理与对齐模块
解决设计稿和实物照片的拍摄角度、尺寸、光照差异
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImageAligner:
    """将实物照片对齐到设计稿的坐标系"""

    def __init__(self, max_features: int = 1000, good_match_percent: float = 0.15):
        self.max_features = max_features
        self.good_match_percent = good_match_percent

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理：统一为RGB三通道。
        输入约定为RGB（app.py 使用 PIL 加载），因此三通道图保持原样，
        不再做 BGR2RGB 转换（否则会调换红蓝通道，污染下游颜色比对）。"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # 三通道已是RGB，保持不变
        return image

    def detect_and_compute(self, image: np.ndarray):
        """检测ORB特征点和描述子"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(self.max_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, des1: np.ndarray, des2: np.ndarray) -> list:
        """匹配特征点，使用汉明距离"""
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(des1, des2, None)
        # 确保是列表（某些版本返回元组）
        matches = list(matches)
        # 按距离排序
        matches.sort(key=lambda x: x.distance)
        # 保留好的匹配
        num_good_matches = int(len(matches) * self.good_match_percent)
        matches = matches[:num_good_matches]
        return matches

    def compute_homography(self, kp1, kp2, matches) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """计算单应性矩阵（透视变换矩阵）"""
        if len(matches) < 4:
            return None, None

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # 使用RANSAC去除异常值
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
        return h, mask

    def align(self, design_image: np.ndarray, photo_image: np.ndarray) -> dict:
        """
        将实物照片对齐到设计稿
        返回包含对齐后的照片和诊断信息的字典
        """
        # 预处理
        design = self.preprocess(design_image)
        photo = self.preprocess(photo_image)

        # 保存原始尺寸
        h_design, w_design = design.shape[:2]
        h_photo, w_photo = photo.shape[:2]

        # 如果图片太大，先缩放到合理尺寸以提高速度
        max_dim = 1500
        scale_design = 1.0
        if max(h_design, w_design) > max_dim:
            scale_design = max_dim / max(h_design, w_design)
            design_small = cv2.resize(design, None, fx=scale_design, fy=scale_design)
        else:
            design_small = design

        scale_photo = 1.0
        if max(h_photo, w_photo) > max_dim:
            scale_photo = max_dim / max(h_photo, w_photo)
            photo_small = cv2.resize(photo, None, fx=scale_photo, fy=scale_photo)
        else:
            photo_small = photo

        # 检测特征点
        kp1, des1 = self.detect_and_compute(design_small)
        kp2, des2 = self.detect_and_compute(photo_small)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            # 特征点不足，回退到简单缩放
            aligned = cv2.resize(photo, (w_design, h_design))
            return {
                "aligned_photo": aligned,
                "success": False,
                "message": "特征点不足，使用简单缩放对齐（精度较低）",
                "match_count": 0,
                "homography": None,
            }

        # 匹配
        matches = self.match_features(des1, des2)
        if len(matches) < 4:
            aligned = cv2.resize(photo, (w_design, h_design))
            return {
                "aligned_photo": aligned,
                "success": False,
                "message": f"匹配点仅{len(matches)}个（需≥4），使用简单缩放对齐",
                "match_count": len(matches),
                "homography": None,
            }

        # 将缩放图上的特征点坐标还原到原始尺寸，直接复用这批匹配，
        # 避免在全尺寸图上重复做一遍 ORB 检测（那会抵消 max_dim 的提速）
        if scale_design != 1.0:
            for kp in kp1:
                kp.pt = (kp.pt[0] / scale_design, kp.pt[1] / scale_design)
        if scale_photo != 1.0:
            for kp in kp2:
                kp.pt = (kp.pt[0] / scale_photo, kp.pt[1] / scale_photo)

        # 用还原到原始坐标的关键点计算单应性矩阵
        h_matrix, mask = self.compute_homography(kp1, kp2, matches)

        if h_matrix is None:
            aligned = cv2.resize(photo, (w_design, h_design))
            return {
                "aligned_photo": aligned,
                "success": False,
                "message": "无法计算透视变换矩阵，使用简单缩放对齐",
                "match_count": len(matches),
                "homography": None,
            }

        # 透视变换：将照片变换到设计稿的坐标系
        aligned = cv2.warpPerspective(photo, h_matrix, (w_design, h_design))

        # 创建匹配可视化图（关键点坐标已还原到原始尺寸）
        match_vis = self._draw_matches(design, photo, kp1, kp2, matches)

        inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0

        return {
            "aligned_photo": aligned,
            "success": True,
            "message": f"成功对齐：{len(matches)}个匹配点，内点率{inlier_ratio:.1%}",
            "match_count": len(matches),
            "homography": h_matrix,
            "match_visualization": match_vis,
            "inlier_ratio": inlier_ratio,
        }

    def _draw_matches(self, img1, img2, kp1, kp2, matches):
        """绘制特征匹配可视化"""
        if len(matches) == 0:
            return None
        # 限制绘制的匹配线数量，避免混乱
        vis_matches = matches[:min(50, len(matches))]
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, vis_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_matches


def detect_edges_for_crop(image: np.ndarray) -> np.ndarray:
    """检测包装边缘并裁剪（可选辅助功能）"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    # 找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image
    # 找最大轮廓
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    # 加一点padding
    padding = 10
    h_img, w_img = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    return image[y1:y2, x1:x2]
