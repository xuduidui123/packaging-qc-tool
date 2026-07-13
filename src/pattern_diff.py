"""
纯图像差异检测（主流程）
========================
彻底舍弃文字识别（OCR），只做"对齐两张图 → 找像素差异 → 把对不上的地方框出来"。
对艺术字 / 外文字符这类 OCR 容易出错的内容尤其稳：不要求机器读懂字，
只要求它看出哪里长得不一样。

流程：
  全局对齐(ORB) → 局部弹性对齐(LocalAligner) → 颜色归一化
  → 容配准误差的 LAB Delta-E 差异 → 边缘容差抑制描边噪声
  → 形态学清理 → 连通域聚合成"差异块" → 按面积×强度排序框出

纯 OpenCV + numpy，不依赖 skimage。文字比对不再参与判定。
"""

import cv2
import numpy as np
from typing import Dict, List, Optional


class PatternDiff:
    """纯图像差异核对：对齐后逐块找差异并高亮。"""

    def __init__(self, delta_e_threshold: float = 18.0,
                 min_area_ratio: float = 0.0008,
                 tolerance_ratio: float = 0.006,
                 edge_tolerance: int = 3,
                 max_regions: int = 40,
                 use_structure: bool = False,
                 structure_gain: float = 2.2):
        """
        delta_e_threshold: LAB 色差阈值下限，越大越不敏感（默认约 18）。
        min_area_ratio:    差异块最小面积占比，滤掉零星噪点。
        tolerance_ratio:   容配准误差的邻域半径占长边比例（吸收残余局部错位）。
        edge_tolerance:    边缘容差（像素），描边噪声在此范围内被抑制。
        max_regions:       最多返回的差异块数量。
        use_structure:     是否叠加结构/纹理差异通道（抓"颜色接近底色、但有无纹理"的差异）。
        structure_gain:    结构通道增益，越大对细纹缺失/多出越敏感（也更易受噪声影响）。
        """
        self.delta_e_threshold = delta_e_threshold
        self.min_area_ratio = min_area_ratio
        self.tolerance_ratio = tolerance_ratio
        self.edge_tolerance = edge_tolerance
        self.max_regions = max_regions
        self.use_structure = use_structure
        self.structure_gain = structure_gain

    # ---------- 预处理 ----------
    def color_normalize(self, design: np.ndarray, photo: np.ndarray) -> np.ndarray:
        """按通道均值/标准差把照片整体亮度色调对齐到设计稿，消除光照差。"""
        result = photo.astype(np.float32)
        for i in range(3):
            d_mean, d_std = float(design[:, :, i].mean()), float(design[:, :, i].std())
            p_mean, p_std = float(photo[:, :, i].mean()), float(photo[:, :, i].std())
            if p_std > 1e-3:
                result[:, :, i] = (result[:, :, i] - p_mean) * (d_std / p_std) + d_mean
        return np.clip(result, 0, 255).astype(np.uint8)

    def package_mask(self, design: np.ndarray) -> np.ndarray:
        """圈出包装本体（排除纯色背景），避免在背景上误报。"""
        h, w = design.shape[:2]
        gray = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (51, 51), 0)
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((41, 41), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < h * w * 0.1:
                continue
            if area > best_area:
                best_area, best = area, cnt
        mask = np.zeros((h, w), np.uint8)
        if best is None:
            mask[:] = 255
            return mask
        x, y, cw, ch = cv2.boundingRect(best)
        # 向内收 2% ：实物在包装物理边缘常有翘边/折痕/透出桌面，
        # 这一圈几乎不承载印刷内容，内缩可根除边缘接缝造成的全宽假高亮。
        ins_x = max(3, int(0.02 * cw))
        ins_y = max(3, int(0.02 * ch))
        cv2.rectangle(mask, (x + ins_x, y + ins_y),
                      (x + cw - ins_x, y + ch - ins_y), 255, -1)
        # 覆盖率过低说明分割失败，回退整幅（去极窄边）
        if float(mask.mean()) / 255.0 < 0.35:
            mask[:] = 255
            m = max(2, int(0.01 * max(h, w)))
            mask[:m, :] = 0; mask[-m:, :] = 0; mask[:, :m] = 0; mask[:, -m:] = 0
        return mask

    # ---------- 差异计算 ----------
    def tolerant_delta_e(self, design: np.ndarray, photo: np.ndarray) -> np.ndarray:
        """
        容配准误差的感知色差图（加权 LAB Delta-E）。
        对每个位置在 ±radius 邻域内取"与设计稿的最小色差"，吸收残余局部错位；
        真正缺失/改动的内容在邻域内找不到匹配，差异被保留。
        """
        h, w = design.shape[:2]
        radius = max(2, int(self.tolerance_ratio * max(h, w)))
        wl, wc = 1.0, 1.5  # 提高色度权重，让偏色/缺印更突出
        lab_d = cv2.cvtColor(design, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_p = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB).astype(np.float32)
        feat_d = np.stack([lab_d[..., 0] * wl, lab_d[..., 1] * wc, lab_d[..., 2] * wc], -1)
        feat_p = np.stack([lab_p[..., 0] * wl, lab_p[..., 1] * wc, lab_p[..., 2] * wc], -1)
        best = np.full((h, w), np.inf, np.float32)
        step = 1 if radius <= 4 else 2
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(feat_p, M, (w, h), flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_REPLICATE)
                d = np.sqrt(np.sum((feat_d - shifted) ** 2, axis=2))
                np.minimum(best, d, out=best)
        return np.clip(best, 0, 255).astype(np.uint8)

    def structure_diff(self, design: np.ndarray, photo: np.ndarray) -> np.ndarray:
        """
        结构/纹理差异通道。
        动机：纯色差通道只看"颜色变了多少"。若某处花纹与底色颜色很接近，
        它在与不在的色差都很小，会被漏检。但只要花纹带一点边缘/肌理，
        它就会在'局部纹理能量'上留下痕迹。本通道比较两图的局部纹理能量密度：
          纹理能量 = 梯度幅值经局部平均后的密度；
          结构差异 = |设计稿纹理能量 - 实物纹理能量|。
        某处设计稿有纹理、实物没有（漏印）→ 能量差大 → 补上色差通道抓不到的差异。
        反之实物多出纹理 → 同样被抓到。

        对错位鲁棒：比较的是'能量密度'而非逐像素边缘，轻微平移下两侧能量近似不变；
        且用邻域最小差进一步吸收残余错位，避免把'只是挪了几像素的同一条纹'误判。

        诚实边界：能量幅值本身仍正比于花纹与底色的对比度。若花纹与底色
        '完全同色同质'（零对比），任何被动成像都无信号可用——这是物理下限，
        本通道只是把'有一点对比但被色差流程滤掉'的细纹救回来，并非无中生有。
        """
        h, w = design.shape[:2]
        gd = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)
        gp = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)

        def energy(gray):
            g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            # 局部平均成"纹理能量密度"，窗口随分辨率自适应
            k = max(7, (int(0.008 * max(h, w)) | 1))
            return cv2.boxFilter(mag, -1, (k, k))

        te_d = energy(gd)
        te_p = energy(gp)

        # 邻域最小差吸收错位：photo 能量在小邻域内取与设计稿最接近的值
        radius = max(2, int(self.tolerance_ratio * max(h, w)))
        best = np.full((h, w), np.inf, np.float32)
        step = 1 if radius <= 4 else 2
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(te_p, M, (w, h), flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_REPLICATE)
                np.minimum(best, np.abs(te_d - shifted), out=best)
        struct = best * self.structure_gain
        return np.clip(struct, 0, 255).astype(np.uint8)

    def edge_suppress(self, design: np.ndarray, diff: np.ndarray) -> np.ndarray:
        """
        边缘容差：设计稿的高对比边缘处，残余亚像素错位最易产生描边假差异。
        在边缘附近**轻度**衰减差异强度，抑制描边光晕。

        注意衰减不能太强：细线艺术字/枝叶等本身就是"细线内容"，整体都落在边缘上。
        若把边缘压得太狠（如衰减到 35%），一旦这类细线内容在实物上整体缺失（漏印），
        其差异信号也会被一并抹掉 → 漏检。因此这里只衰减到 70%，
        真正的错位光晕主要交给"容配准误差的邻域最小 ΔE"和后续模板核验去处理，
        本步仅做补充抑制，保住细线缺失的召回。
        """
        gray = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        if self.edge_tolerance > 0:
            k = 2 * self.edge_tolerance + 1
            edges = cv2.dilate(edges, np.ones((k, k), np.uint8), iterations=1)
        atten = np.where(edges > 0, 0.70, 1.0).astype(np.float32)
        return (diff.astype(np.float32) * atten).astype(np.uint8)

    def aggregate_regions(self, diff: np.ndarray, mask: np.ndarray,
                          design: np.ndarray, photo: np.ndarray) -> List[Dict]:
        """把差异图聚合成差异块：阈值 → 形态学 → 连通域 → 排序。"""
        h, w = diff.shape
        min_area = max(40, int(h * w * self.min_area_ratio))
        d = cv2.bitwise_and(diff, diff, mask=mask)

        # 去超大范围渐变（残余光照/阴影），保留中等尺寸色块
        k = max(51, (min(h, w) // 4) | 1)
        low = cv2.GaussianBlur(d.astype(np.float32), (k, k), 0)
        high = np.clip(d.astype(np.float32) - low * 0.6, 0, 255).astype(np.uint8)

        if int(high.max()) < 8:
            return []
        otsu = cv2.threshold(high, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        thr = max(otsu, self.delta_e_threshold)
        _, binary = cv2.threshold(high, thr, 255, cv2.THRESH_BINARY)

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), 2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, wr, hr = cv2.boundingRect(cnt)
            if wr < 8 or hr < 8:
                continue
            if wr > 0.95 * w and hr > 0.95 * h:
                continue
            # 贴边细长条多为翘边/背景残留；边距按比例（外圈~1.5%）判定
            mx = max(3, int(0.015 * w))
            my = max(3, int(0.015 * h))
            if (y <= my or y + hr >= h - my) and hr < 0.05 * h:
                continue
            if (x <= mx or x + wr >= w - mx) and wr < 0.05 * w:
                continue
            # 近全宽的横向细带（且落在画面上/下 20%）：多为实物与桌面/背景的
            # 物理接缝，非印刷差异。设计稿上真正整行缺印极少呈均匀实心细带，故过滤。
            near_top = y + hr < 0.2 * h
            near_bottom = y > 0.8 * h
            if wr > 0.6 * w and hr < 0.12 * h and (near_top or near_bottom):
                continue
            roi = d[y:y+hr, x:x+wr]
            regions.append({
                "bbox": (int(x), int(y), int(wr), int(hr)),
                "area": float(area),
                "avg_diff": float(roi.mean()),
                "peak_diff": float(np.percentile(roi, 90)),
            })
        regions.sort(key=lambda r: r["area"] * r["peak_diff"], reverse=True)
        return regions[:self.max_regions]

    def _merge_regions(self, regions: List[Dict]) -> List[Dict]:
        """合并来自色差/结构两通道的候选框：重叠(IoU 高或互相包含中心)的去重保强。"""
        def iou(a, b):
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            x1, y1 = max(ax, bx), max(ay, by)
            x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter == 0:
                return 0.0
            return inter / float(aw * ah + bw * bh - inter)
        regions = sorted(regions, key=lambda r: r["area"] * r["peak_diff"], reverse=True)
        kept = []
        for r in regions:
            x, y, w, h = r["bbox"]; cx, cy = x + w / 2, y + h / 2
            dup = False
            for k in kept:
                kx, ky, kw, kh = k["bbox"]
                inside = kx <= cx <= kx + kw and ky <= cy <= ky + kh
                if iou(r["bbox"], k["bbox"]) > 0.3 or inside:
                    dup = True; break
            if not dup:
                kept.append(r)
        return kept[:self.max_regions]

    def verify_regions(self, design: np.ndarray, photo: np.ndarray,
                       regions: List[Dict]) -> List[Dict]:
        """
        二次核验：把设计稿小块在实物更大邻域内模板搜索。
        结构高度吻合且该处颜色也一致 => 内容其实存在只是错位 => 丢弃；
        找不到吻合(缺失/移位) 或吻合处颜色仍不同(改色) => 真差异 => 保留。
        """
        h, w = design.shape[:2]
        gd = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)
        gp = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        search = max(10, int(0.02 * max(h, w)))
        kept = []
        for r in regions:
            x, y, wr, hr = r["bbox"]
            patch = gd[y:y+hr, x:x+wr]
            y0, x0 = max(0, y - search), max(0, x - search)
            y1, x1 = min(h, y + hr + search), min(w, x + wr + search)
            win = gp[y0:y1, x0:x1]
            if patch.shape[0] < 4 or patch.shape[1] < 4 or \
               win.shape[0] < patch.shape[0] or win.shape[1] < patch.shape[1]:
                kept.append(r); continue
            res = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            bx, by = x0 + maxloc[0], y0 + maxloc[1]
            pd = cv2.cvtColor(cv2.GaussianBlur(design[y:y+hr, x:x+wr], (0, 0), 2.0),
                              cv2.COLOR_RGB2LAB).astype(np.float32)
            pp = cv2.cvtColor(cv2.GaussianBlur(photo[by:by+hr, bx:bx+wr], (0, 0), 2.0),
                              cv2.COLOR_RGB2LAB).astype(np.float32)
            per_pixel = np.sqrt(np.sum((pd - pp) ** 2, axis=2))
            color_res = float(np.percentile(per_pixel, 90))
            if maxv >= 0.70 and color_res < 22.0:
                continue  # 内容存在且低频颜色一致 => 仅错位 => 丢弃
            kept.append(r)
        return kept

    # ---------- 可视化 ----------
    def visualize(self, design: np.ndarray, photo: np.ndarray,
                  diff: np.ndarray, regions: List[Dict]) -> Dict[str, np.ndarray]:
        highlight = photo.copy()
        for i, r in enumerate(regions):
            x, y, wr, hr = r["bbox"]
            cv2.rectangle(highlight, (x, y), (x + wr, y + hr), (255, 0, 0), 3)
            roi = highlight[y:y+hr, x:x+wr].copy()
            red = np.zeros_like(roi); red[:] = (255, 0, 0)
            highlight[y:y+hr, x:x+wr] = cv2.addWeighted(roi, 0.75, red, 0.25, 0)
            cv2.putText(highlight, str(i + 1), (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        side = np.hstack([design, photo, highlight])
        # 红绿叠加图：设计稿入绿通道、实物入红通道，对齐处呈黄，差异处露单色
        overlay = np.zeros_like(design)
        overlay[:, :, 1] = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)
        overlay[:, :, 0] = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        binary = (diff > self.delta_e_threshold).astype(np.uint8) * 255
        return {"diff_highlight": highlight, "heatmap": heat, "side_by_side": side,
                "red_green_overlay": overlay, "binary_diff": binary}

    # ---------- 主入口 ----------
    def check(self, design: np.ndarray, aligned_photo: np.ndarray) -> Dict:
        """
        输入：设计稿、已（全局+局部）对齐的实物照片，尺寸一致。
        返回：差异块列表、差异图、可视化、判定。
        """
        h, w = design.shape[:2]
        if aligned_photo.shape[:2] != (h, w):
            aligned_photo = cv2.resize(aligned_photo, (w, h))

        mask = self.package_mask(design)
        # 排除对齐后的黑边（透视/remap 填充区）
        gray_ap = cv2.cvtColor(aligned_photo, cv2.COLOR_RGB2GRAY)
        valid = (gray_ap > 8).astype(np.uint8) * 255
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8), 1)
        mask = cv2.bitwise_and(mask, valid)

        photo_n = self.color_normalize(design, aligned_photo)
        # 轻微模糊压制亚像素描边
        d_cmp = cv2.GaussianBlur(design, (3, 3), 0)
        p_cmp = cv2.GaussianBlur(photo_n, (3, 3), 0)

        color_diff = self.edge_suppress(design, self.tolerant_delta_e(d_cmp, p_cmp))
        color_diff = cv2.bitwise_and(color_diff, color_diff, mask=mask)

        # 两个通道**各自独立**检测后取并集，而非先相加再统一阈值——
        # 否则结构通道的宽带纹理能量会抬高 Otsu 自适应阈值，反而压掉真实的纯色差差异。
        regions = self.aggregate_regions(color_diff, mask, design, photo_n)
        diff = color_diff
        if self.use_structure:
            struct = self.edge_suppress(design, self.structure_diff(d_cmp, p_cmp))
            struct = cv2.bitwise_and(struct, struct, mask=mask)
            struct_regions = self.aggregate_regions(struct, mask, design, photo_n)
            regions = self._merge_regions(regions + struct_regions)
            diff = np.maximum(color_diff, struct)  # 仅用于热图可视化

        regions = self.verify_regions(design, photo_n, regions)

        visuals = self.visualize(design, photo_n, diff, regions)

        # 相似度统计（替代 SSIM）：mask 内的平均 ΔE，及"色差达标像素占比"。
        m = mask > 0
        if int(m.sum()) > 0:
            mean_delta_e = float(diff[m].mean())
            match_rate = float((diff[m] <= self.delta_e_threshold).mean())
        else:
            mean_delta_e, match_rate = 0.0, 1.0

        issues = []
        if regions:
            issues.append(f"发现 {len(regions)} 个显著差异区域")
        return {
            "regions": regions,
            "diff_map": diff,
            "visualizations": visuals,
            "issues": issues,
            "passed": len(regions) == 0,
            "package_mask": mask,
            "match_rate": match_rate,      # 0-1，越高越一致
            "mean_delta_e": mean_delta_e,  # 平均感知色差
        }
