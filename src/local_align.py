"""
局部（弹性）对齐模块
=====================
全局 ORB 单应变换只能把包装整体摆正，但实物是会翘边/鼓起的曲面，
一张单应矩阵压不平局部拉伸。文字这种高频细节对亚像素错位极敏感，
残余错位会在纯像素比对里把每个字的边缘都描成红边 —— 满屏假高亮。

本模块在全局对齐之后再做一层"分块光流式"局部配准：
  1. 把设计稿划成网格；
  2. 每个网格块在实物的小邻域内做模板匹配，得到该块的局部位移；
  3. 用结构相似度/匹配置信度给位移加权，剔除不可信块；
  4. 把稀疏位移插值成平滑稠密位移场；
  5. 用位移场 remap 实物照片，使其逐块贴合设计稿。

产出的位移场同时可用来生成"对齐残差图"，供上层判断哪些地方
根本对不上（可能是真缺失/移位），哪些只是被顺利拉正了。
纯 OpenCV + numpy 实现，不依赖 skimage。
"""

import cv2
import numpy as np
from typing import Tuple


class LocalAligner:
    """在全局对齐后做分块弹性配准，压制曲面翘曲造成的局部错位。"""

    def __init__(self, grid: int = 24, search_ratio: float = 0.015,
                 min_conf: float = 0.35):
        """
        grid:         网格密度（横向块数，纵向按长宽比自动缩放）。越大越精细但越慢。
        search_ratio: 每块搜索半径占图像长边的比例（局部位移的最大幅度）。
        min_conf:     模板匹配置信度(NCC)下限，低于此的块位移不可信，丢弃后靠插值补。
        """
        self.grid = grid
        self.search_ratio = search_ratio
        self.min_conf = min_conf

    def _block_shifts(self, design_gray: np.ndarray, photo_gray: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """逐块模板匹配，返回块中心坐标、位移(dx,dy)与置信度。"""
        h, w = design_gray.shape
        search = max(6, int(self.search_ratio * max(h, w)))
        nx = self.grid
        ny = max(4, int(round(self.grid * h / w)))
        bw, bh = w // nx, h // ny
        # 块要比搜索半径大，否则模板太小匹配不稳
        bw = max(bw, 3 * search)
        bh = max(bh, 3 * search)

        centers, shifts, confs = [], [], []
        for gy in range(ny):
            for gx in range(nx):
                cx = int((gx + 0.5) * w / nx)
                cy = int((gy + 0.5) * h / ny)
                x0, y0 = cx - bw // 2, cy - bh // 2
                x1, y1 = x0 + bw, y0 + bh
                if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                    continue
                templ = design_gray[y0:y1, x0:x1]
                # 低纹理块（纯色背景）无从匹配，跳过 —— 位移交给插值
                if float(templ.std()) < 6.0:
                    continue
                sx0, sy0 = max(0, x0 - search), max(0, y0 - search)
                sx1, sy1 = min(w, x1 + search), min(h, y1 + search)
                win = photo_gray[sy0:sy1, sx0:sx1]
                if win.shape[0] < templ.shape[0] or win.shape[1] < templ.shape[1]:
                    continue
                res = cv2.matchTemplate(win, templ, cv2.TM_CCOEFF_NORMED)
                _, maxv, _, maxloc = cv2.minMaxLoc(res)
                if maxv < self.min_conf:
                    continue
                # 匹配到的位置相对原位置的偏移
                found_x = sx0 + maxloc[0]
                found_y = sy0 + maxloc[1]
                dx = found_x - x0
                dy = found_y - y0
                # 限幅：超过搜索半径的多半是误匹配
                if abs(dx) > search or abs(dy) > search:
                    continue
                centers.append((cx, cy))
                shifts.append((dx, dy))
                confs.append(maxv)

        return (np.array(centers, np.float32),
                np.array(shifts, np.float32),
                np.array(confs, np.float32))

    def _dense_field(self, centers, shifts, confs, shape) -> Tuple[np.ndarray, np.ndarray]:
        """把稀疏块位移插值成稠密位移场 (map dx, dy)，并做平滑。"""
        h, w = shape
        if len(centers) < 4:
            return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)

        # 反距离加权（IDW）散点插值 —— 无 scipy 依赖，稳且平滑。
        # 为控制耗时，在降采样网格上插值，再放大回原尺寸。
        gs = 64
        ys = np.linspace(0, h - 1, gs)
        xs = np.linspace(0, w - 1, gs)
        gx, gy = np.meshgrid(xs, ys)
        fdx = np.zeros((gs, gs), np.float32)
        fdy = np.zeros((gs, gs), np.float32)
        cx = centers[:, 0][None, None, :]
        cy = centers[:, 1][None, None, :]
        sdx = shifts[:, 0][None, None, :]
        sdy = shifts[:, 1][None, None, :]
        wconf = confs[None, None, :]
        d2 = (gx[..., None] - cx) ** 2 + (gy[..., None] - cy) ** 2 + 1.0
        wts = wconf / d2  # 越近、越可信权重越高
        wsum = np.sum(wts, axis=2) + 1e-6
        fdx = np.sum(wts * sdx, axis=2) / wsum
        fdy = np.sum(wts * sdy, axis=2) / wsum
        # 放大到原尺寸并高斯平滑，得到连续位移场
        fdx = cv2.resize(fdx.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        fdy = cv2.resize(fdy.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        k = max(31, (min(h, w) // 20) | 1)
        fdx = cv2.GaussianBlur(fdx, (k, k), 0)
        fdy = cv2.GaussianBlur(fdy, (k, k), 0)
        return fdx, fdy

    def align(self, design: np.ndarray, photo: np.ndarray) -> dict:
        """
        输入已全局对齐、尺寸一致的设计稿与实物照片（RGB）。
        返回：局部对齐后的照片、位移场幅度图、可信块数量。
        """
        h, w = design.shape[:2]
        if photo.shape[:2] != (h, w):
            photo = cv2.resize(photo, (w, h))
        gd = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)
        gp = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)

        centers, shifts, confs = self._block_shifts(gd, gp)
        fdx, fdy = self._dense_field(centers, shifts, confs, (h, w))

        # 构造 remap 坐标：photo 中 (x+dx, y+dy) 的像素搬到设计稿的 (x,y)
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                   np.arange(h, dtype=np.float32))
        remap_x = (map_x + fdx).astype(np.float32)
        remap_y = (map_y + fdy).astype(np.float32)
        warped = cv2.remap(photo, remap_x, remap_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

        magnitude = np.sqrt(fdx ** 2 + fdy ** 2)
        return {
            "aligned_photo": warped,
            "displacement_magnitude": magnitude,
            "block_count": int(len(centers)),
            "max_shift": float(magnitude.max()) if magnitude.size else 0.0,
        }
