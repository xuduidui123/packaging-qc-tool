"""
图案差异检测模块 (改进版)
使用SSIM结构相似度和像素级差异检测图案问题
改进点：
1. 颜色归一化 - 消除整体光照/色调差异
2. 包装区域自动提取 - 只比对包装本体，排除背景
3. 自适应差异检测 - 形态学处理分离差异区域
4. 局部差异增强 - 突出小面积图案缺失
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim


class PatternChecker:
    """包装图案核对：检测图案缺失、偏移、色差等问题"""

    def __init__(self, ssim_threshold: float = 0.92, pixel_threshold: int = 20,
                 min_area_ratio: float = 0.001):
        self.ssim_threshold = ssim_threshold
        self.pixel_threshold = pixel_threshold
        self.min_area_ratio = min_area_ratio  # 最小区域占图像面积的比例

    def color_normalize(self, design: np.ndarray, photo: np.ndarray) -> np.ndarray:
        """
        颜色归一化：让照片的整体亮度/色调接近设计稿
        使用均值和标准差对齐（简单高效）
        """
        result = photo.astype(np.float32)
        for i in range(3):
            d_mean = np.mean(design[:, :, i])
            d_std = np.std(design[:, :, i])
            p_mean = np.mean(photo[:, :, i])
            p_std = np.std(photo[:, :, i])
            if p_std > 0:
                result[:, :, i] = (result[:, :, i] - p_mean) * (d_std / p_std) + d_mean
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def extract_content_mask(self, design: np.ndarray) -> np.ndarray:
        """
        从设计稿中提取'有内容'的区域（图案、文字等），而非包装外边框
        方法：背景通常是纯色，内容与背景有明显颜色差异
        """
        h, w = design.shape[:2]
        gray = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)

        # --- 通道A：灰度高频细节（捕捉深色线条/文字）---
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        detail = cv2.absdiff(gray, blurred)
        _, mask_detail = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- 通道B：颜色偏离背景（捕捉'亮度接近但有颜色'的淡彩图案）---
        # 背景通常是近中性的主色调；有颜色的图案在 LAB 的 a/b 上偏离背景
        lab = cv2.cvtColor(design, cv2.COLOR_RGB2LAB).astype(np.float32)
        a_bg = np.median(lab[:, :, 1])
        b_bg = np.median(lab[:, :, 2])
        chroma_dev = np.sqrt((lab[:, :, 1] - a_bg) ** 2 + (lab[:, :, 2] - b_bg) ** 2)
        chroma_dev = np.clip(chroma_dev, 0, 255).astype(np.uint8)
        # Otsu 自适应阈值，但设一个下限避免把噪声当内容
        c_thr = cv2.threshold(chroma_dev, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        _, mask_chroma = cv2.threshold(chroma_dev, max(c_thr, 8), 255, cv2.THRESH_BINARY)

        # 合并：灰度细节 或 颜色偏离，都算作内容
        mask = cv2.bitwise_or(mask_detail, mask_chroma)

        # 形态学闭运算：连接相邻内容，填充小孔
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 适度膨胀，确保淡彩图案边缘被完整纳入检测区
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 小区域过滤（可能是噪点）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 20:
                cv2.drawContours(mask, [cnt], -1, 0, -1)

        return mask

    def detect_barcode_mask(self, image: np.ndarray) -> np.ndarray:
        """
        检测条形码区域（密集竖条）。返回条码区域为 255 的掩码。
        条码本应通过扫码解码来验证，用像素比对只会因微小错位产生大量假差异，
        因此在图案核对中将其整体排除。
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        # 竖直方向梯度强、水平方向梯度弱 => 条码条纹的典型特征
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
        grad = cv2.convertScaleAbs(cv2.subtract(np.abs(gx), np.abs(gy)))
        grad = cv2.blur(grad, (9, 9))
        _, th = cv2.threshold(grad, 150, 255, cv2.THRESH_BINARY)
        # 横向闭运算把条纹连成整块
        kx = max(15, w // 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, max(5, h // 120)))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed, np.ones((5, 5), np.uint8), iterations=2)
        # 饱和度：条码是黑白的（低饱和），据此排除彩色网格/色卡等误检
        if len(image.shape) == 3:
            sat = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]
        else:
            sat = np.zeros((h, w), dtype=np.uint8)

        mask = np.zeros((h, w), dtype=np.uint8)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            # 条码通常是较宽的横向区块，且占一定面积
            if area < 0.01 * h * w:
                continue
            if cw < 1.5 * ch:  # 条码整体明显偏横向（一排竖条）
                continue
            # 低饱和度校验：条码为黑白，彩色内容（色卡/图案）会被排除
            if float(np.mean(sat[y:y + ch, x:x + cw])) > 45:
                continue
            pad = int(0.02 * max(h, w))
            cv2.rectangle(mask, (max(0, x - pad), max(0, y - pad)),
                          (min(w, x + cw + pad), min(h, y + ch + pad)), 255, -1)
        return mask

    def extract_package_mask(self, design: np.ndarray) -> np.ndarray:
        """
        从设计稿中提取包装本体区域（外边框）
        策略：对设计稿做大核模糊 + Otsu阈值，只保留整体包装区域
        """
        h, w = design.shape[:2]
        gray = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)

        # 大核高斯模糊：抹平内部细节，保留整体包装区域
        blurred = cv2.GaussianBlur(gray, (51, 51), 0)

        # Otsu自动阈值分割背景和前景（包装区域）
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 大核闭运算：填充内部空洞，连接前景
        kernel = np.ones((41, 41), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 找轮廓，找面积最大的矩形外接轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return np.ones((h, w), dtype=np.uint8)

        # 找面积最大且接近矩形、宽高比合理的轮廓
        best = None
        best_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < h * w * 0.1:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            rect_area = cw * ch
            if rect_area == 0:
                continue
            rect_ratio = area / rect_area
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            # 包装通常是横向矩形，宽高比>1.5
            if aspect < 1.2:
                continue
            score = area * rect_ratio
            if score > best_score:
                best_score = score
                best = cnt

        if best is None:
            return np.ones((h, w), dtype=np.uint8)

        x, y, cw, ch = cv2.boundingRect(best)
        padding = 3
        x1 = max(0, x + padding)
        y1 = max(0, y + padding)
        x2 = min(w, x + cw - padding)
        y2 = min(h, y + ch - padding)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def compute_ssim(self, design: np.ndarray, photo: np.ndarray) -> Tuple[float, np.ndarray]:
        """计算两图的结构相似度"""
        if len(design.shape) == 3:
            gray1 = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = design
        if len(photo.shape) == 3:
            gray2 = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = photo

        score, diff = ssim(gray1, gray2, full=True)
        return score, diff

    def compute_pixel_diff(self, design: np.ndarray, photo: np.ndarray) -> np.ndarray:
        """计算像素级绝对差异"""
        diff = cv2.absdiff(design, photo)
        return diff

    def tolerant_color_diff(self, design: np.ndarray, photo: np.ndarray,
                            radius: int = None, step: int = 2) -> np.ndarray:
        """
        容配准误差的感知色差图。
        动机：设计稿是平面，实物是会翘边的曲面，单一透视矩阵无法让整幅图
        逐像素对齐——每个图案边缘都会有几像素局部错位。若逐像素直接比对，
        这些错位会把所有图案都误判成差异，淹没真正的缺印/改色。
        做法：对每个位置，在 ±radius 的小邻域内取'与设计稿的最小色差'，
        从而吸收配准误差；而真正缺失/改动的内容在邻域内找不到匹配、差异保留。
        返回：0-255 单通道差异强度图（LAB Delta-E，加权色度）。
        """
        if len(design.shape) != 3 or len(photo.shape) != 3:
            return np.zeros(design.shape[:2], dtype=np.uint8)
        h, w = design.shape[:2]
        if radius is None:
            radius = max(4, int(0.007 * max(h, w)))  # 随分辨率自适应（约0.7%，吸收旋转标签的对齐残差）
        wl, wc = 1.0, 1.5
        lab_d = cv2.cvtColor(design, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_p = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB).astype(np.float32)
        # 预乘通道权重，之后用欧氏距离即为加权 Delta-E
        feat_d = np.stack([lab_d[:, :, 0] * wl, lab_d[:, :, 1] * wc, lab_d[:, :, 2] * wc], -1)
        feat_p = np.stack([lab_p[:, :, 0] * wl, lab_p[:, :, 1] * wc, lab_p[:, :, 2] * wc], -1)
        best = np.full((h, w), np.inf, dtype=np.float32)
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                # 平移 photo 特征（边界复制，避免引入假边）
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(feat_p, M, (w, h), flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_REPLICATE)
                d = np.sqrt(np.sum((feat_d - shifted) ** 2, axis=2))
                np.minimum(best, d, out=best)
        return np.clip(best, 0, 255).astype(np.uint8)

    def compute_color_diff(self, design: np.ndarray, photo: np.ndarray) -> np.ndarray:
        """
        感知色差图（LAB 空间的 Delta-E，逐像素）。
        返回：0-255 的单通道色差强度图。
        """
        if len(design.shape) != 3 or len(photo.shape) != 3:
            return np.zeros(design.shape[:2], dtype=np.uint8)
        lab1 = cv2.cvtColor(design, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB).astype(np.float32)
        # 提高 a/b 色度通道权重：让颜色差异（而非亮度）更突出
        wl, wc = 1.0, 1.5
        dl = (lab1[:, :, 0] - lab2[:, :, 0]) * wl
        da = (lab1[:, :, 1] - lab2[:, :, 1]) * wc
        db = (lab1[:, :, 2] - lab2[:, :, 2]) * wc
        delta = np.sqrt(dl ** 2 + da ** 2 + db ** 2)
        delta = np.clip(delta, 0, 255).astype(np.uint8)
        return delta

    def find_difference_regions(self, diff_map: np.ndarray,
                                 pixel_diff: np.ndarray,
                                 mask: Optional[np.ndarray] = None,
                                 max_regions: int = 30,
                                 color_diff: Optional[np.ndarray] = None) -> list:
        """
        从差异图中找出显著差异区域
        关键改进：SSIM灰度差 + 像素差 + LAB色差 三者取max，
        再用Otsu自动阈值分割，避免'只有颜色差'的差异被漏掉。
        """
        h, w = diff_map.shape
        min_area = max(30, int(h * w * self.min_area_ratio))

        # 1. SSIM差异图归一化：skimage 的 S 图 1=相同、0=不同，
        #    取 (1 - S) 才是"差异强度"（此前用反了，靠高通侥幸抵消）
        diff_norm = np.clip((1.0 - diff_map) * 255, 0, 255).astype(np.uint8)
        # 2. 像素差图（灰度）
        if len(pixel_diff.shape) == 3:
            pixel_gray = cv2.cvtColor(pixel_diff, cv2.COLOR_RGB2GRAY)
        else:
            pixel_gray = pixel_diff

        # 合并差异图。
        # 若提供了'容配准误差'的差异图（color_diff），以它为主信号——
        # 因为逐像素的 SSIM/像素差在曲面实物上会被局部错位刷爆、淹没真实差异。
        # 该图已同时编码亮度与颜色差异，且吸收了几像素的配准误差。
        if color_diff is not None:
            combined = color_diff
        else:
            combined = np.maximum(diff_norm, pixel_gray)

        # 应用mask
        if mask is not None:
            combined = cv2.bitwise_and(combined, combined, mask=mask)

        # 3. 高通滤波：只减去'超大范围'的渐变光照/阴影，保留中等尺寸色块。
        #    核必须足够大——否则会把大面积纯色差异（如整朵花改色、大 logo 缺印）
        #    的内部抹平，只剩边缘，导致这类差异被漏检。核随图像尺寸自适应。
        k = max(51, (min(h, w) // 4) | 1)  # 保证为奇数
        blurred = cv2.GaussianBlur(combined.astype(np.float32), (k, k), 0)
        highpass = combined.astype(np.float32) - blurred
        highpass = np.clip(highpass, 0, 255).astype(np.uint8)

        # 4. 用Otsu自动阈值找到最佳分割点（不是固定阈值！）
        # Otsu会找到最佳阈值，把高差异区域和低差异区域分开
        if np.max(highpass) < 10:
            return []  # 几乎无差异

        otsu_thresh = cv2.threshold(highpass, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        # Otsu阈值可能过低，设一个下限；下限绑定到 pixel_threshold，
        # 让 UI 滑块能真正调节区域检测灵敏度（默认30可压掉噪声级差异）
        floor = max(10, self.pixel_threshold)
        effective_thresh = max(otsu_thresh, floor)
        _, binary = cv2.threshold(highpass, effective_thresh, 255, cv2.THRESH_BINARY)

        # 5. 形态学处理：先开运算去噪点，再闭运算连接断裂
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # 6. 分水岭分割：如果连通区域太大，尝试分割内部差异
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w_r, h_r = cv2.boundingRect(cnt)
            if w_r < 8 or h_r < 8:
                continue
            if w_r > w * 0.95 and h_r > h * 0.95:
                continue
            # 过滤'贴着图像边缘的细长条'：多为实物翘边/背景残留，非印刷缺陷
            margin = 3
            touches_top = y <= margin
            touches_bottom = y + h_r >= h - margin
            touches_left = x <= margin
            touches_right = x + w_r >= w - margin
            if (touches_top or touches_bottom) and h_r < 0.04 * h:
                continue
            if (touches_left or touches_right) and w_r < 0.04 * w:
                continue

            # 计算该区域在原始差异图中的平均强度
            roi = combined[y:y+h_r, x:x+w_r]
            if mask is not None:
                roi_mask = mask[y:y+h_r, x:x+w_r]
                masked_pixels = np.sum(roi_mask > 0)
                if masked_pixels > 0:
                    avg_diff = float(np.sum(roi[roi_mask > 0]) / masked_pixels)
                else:
                    avg_diff = 0
            else:
                avg_diff = float(np.mean(roi))

            # 计算局部差异强度（高通图）
            roi_high = highpass[y:y+h_r, x:x+w_r]
            local_diff = float(np.mean(roi_high))

            regions.append({
                "bbox": (x, y, w_r, h_r),
                "area": area,
                "avg_diff": avg_diff,
                "local_diff": local_diff,
            })

        # 按面积和局部差异综合排序
        regions.sort(key=lambda r: r["area"] * r.get("local_diff", r["avg_diff"]), reverse=True)
        return regions[:max_regions]

    def verify_regions(self, design: np.ndarray, photo: np.ndarray, regions: list,
                       ncc_thresh: float = 0.70, color_thresh: float = 25.0,
                       search: int = None) -> list:
        """
        对候选差异区域做二次核验，剔除'仅因对齐错位'造成的假差异（条码/图标/文字边缘等）。
        原理：把设计稿的该小块在实物的更大邻域内模板搜索——
          · 若能找到结构高度吻合(NCC高)且该处颜色也一致的位置 => 内容其实存在，只是错位 => 丢弃；
          · 若找不到吻合(缺失/移位) 或 吻合处颜色仍不同(改色) => 真差异 => 保留。
        因此对'漏印/改色'保持敏感，对'高频细节错位'鲁棒。
        """
        h, w = design.shape[:2]
        gd = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)
        gp = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        lab_d = cv2.cvtColor(design, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_p = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB).astype(np.float32)
        if search is None:
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
                kept.append(r)
                continue
            res = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            bx, by = x0 + maxloc[0], y0 + maxloc[1]
            # 残差在'低通(模糊)'后计算：
            #   · 细线/文字/图标的错位残差集中在边缘(高频)，模糊后相互抵消 => 残差低 => 判为错位；
            #   · 真实的缺印/改色/黑↔灰是整块(低频)差异，模糊后依然存在 => 残差高 => 保留。
            pd = cv2.cvtColor(cv2.GaussianBlur(design[y:y+hr, x:x+wr], (0, 0), 2.0),
                              cv2.COLOR_RGB2LAB).astype(np.float32)
            pp = cv2.cvtColor(cv2.GaussianBlur(photo[by:by+hr, bx:bx+wr], (0, 0), 2.0),
                              cv2.COLOR_RGB2LAB).astype(np.float32)
            diff = pd - pp
            per_pixel = np.sqrt(np.sum(diff ** 2, axis=2))
            # 用高分位(峰值)而非平均：大区域里只要有一小块真实差异也不会被白底稀释掉
            color_res = float(np.percentile(per_pixel, 90))
            if maxv >= ncc_thresh and color_res < color_thresh:
                continue  # 内容存在且低频亮度/颜色一致 => 仅错位 => 丢弃
            kept.append(r)
        return kept

    def _drop_barcode_bars(self, photo: np.ndarray, design: np.ndarray, regions: list) -> list:
        """剔除条码竖条类假差异：细高（高/宽比大）、黑白（低饱和）的窄条。
        真实印刷缺陷极少呈现为这种纯黑细长条，故可安全过滤。"""
        h, w = photo.shape[:2]
        sat_p = cv2.cvtColor(photo, cv2.COLOR_RGB2HSV)[:, :, 1] if len(photo.shape) == 3 else None
        sat_d = cv2.cvtColor(design, cv2.COLOR_RGB2HSV)[:, :, 1] if len(design.shape) == 3 else None
        kept = []
        for r in regions:
            x, y, wr, hr = r["bbox"]
            thin = wr <= max(6, 0.025 * w)
            tall = hr >= 3.0 * max(wr, 1)
            low_sat = True
            if sat_p is not None:
                mp = float(np.mean(sat_p[y:y+hr, x:x+wr]))
                md = float(np.mean(sat_d[y:y+hr, x:x+wr]))
                low_sat = mp < 45 and md < 45
            if thin and tall and low_sat:
                continue  # 条码竖条
            kept.append(r)
        return kept

    def create_diff_visualization(self, design: np.ndarray, photo: np.ndarray,
                                   diff_map: np.ndarray,
                                   regions: list,
                                   mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """创建多种差异可视化图"""
        h, w = design.shape[:2]

        # 1. 红绿叠加图
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        if len(design.shape) == 3:
            gray_design = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)
        else:
            gray_design = design
        if len(photo.shape) == 3:
            gray_photo = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        else:
            gray_photo = photo

        overlay[:, :, 1] = gray_design
        overlay[:, :, 2] = gray_photo
        overlay[:, :, 0] = 0

        # 2. 差异热图（带mask）：同样取 (1 - S) 作为差异强度
        diff_norm = np.clip((1.0 - diff_map) * 255, 0, 255).astype(np.uint8)
        if mask is not None:
            diff_norm_masked = cv2.bitwise_and(diff_norm, diff_norm, mask=mask)
        else:
            diff_norm_masked = diff_norm
        heatmap = cv2.applyColorMap(diff_norm_masked, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 3. 差异高亮（在normalized照片上标注）
        diff_mask = photo.copy()
        for r in regions:
            x, y, w_r, h_r = r["bbox"]
            cv2.rectangle(diff_mask, (x, y), (x + w_r, y + h_r), (255, 0, 0), 3)
            # 半透明红色填充
            overlay_rect = diff_mask[y:y+h_r, x:x+w_r].copy()
            red_fill = np.zeros_like(overlay_rect)
            red_fill[:, :] = (255, 0, 0)
            alpha = 0.25
            diff_mask[y:y+h_r, x:x+w_r] = cv2.addWeighted(overlay_rect, 1-alpha, red_fill, alpha, 0)

        # 4. 并排对比
        comparison = np.hstack([design, photo, diff_mask])

        # 5. 差异掩码图（二值化差异）
        _, binary = cv2.threshold(diff_norm_masked, self.pixel_threshold, 255, cv2.THRESH_BINARY)

        return {
            "red_green_overlay": overlay,
            "heatmap": heatmap,
            "diff_highlight": diff_mask,
            "side_by_side": comparison,
            "binary_diff": binary,
        }

    def check(self, design: np.ndarray, aligned_photo: np.ndarray) -> Dict:
        """执行图案核对（改进版）"""
        # 确保尺寸一致
        h, w = design.shape[:2]
        aligned_photo = cv2.resize(aligned_photo, (w, h))

        # 1. 检测区域 = 包装本体（package_mask）。
        #    注意：不再用 content_mask 去 AND 限制比对区域——那样会把
        #    '亮度低、对比弱'的淡彩图案排除在外，导致缺印类差异永远漏检。
        #    背景在两图中一致、差异≈0 不会造成误报，所以只需圈住包装本体即可。
        package_mask = self.extract_package_mask(design)
        content_mask = self.extract_content_mask(design)  # 仅用于返回/可视化参考
        # 覆盖率过低说明 package_mask 分割失败（如整幅低对比的包装稿被误分割），
        # 此时回退为整幅画面——包装稿通常铺满画面，背景一致不会造成误报，
        # 但若继续只用那一小块会漏掉画面其他位置的真实差异（如多出的色块）。
        if package_mask is not None and float(np.mean(package_mask > 0)) >= 0.40:
            detection_mask = package_mask.copy()
        else:
            # 整幅画面，去掉极窄边缘
            detection_mask = np.ones((h, w), dtype=np.uint8) * 255
            m = max(2, int(0.01 * max(h, w)))
            detection_mask[:m, :] = 0; detection_mask[-m:, :] = 0
            detection_mask[:, :m] = 0; detection_mask[:, -m:] = 0

        # 1b. 排除对齐照片的黑色边框区域（warpPerspective 透视变换后
        #     照片外的区域会被填成黑色，若不排除会与设计稿产生巨大假差异）
        gray_ap = cv2.cvtColor(aligned_photo, cv2.COLOR_RGB2GRAY) if len(aligned_photo.shape) == 3 else aligned_photo
        valid_mask = (gray_ap > 8).astype(np.uint8) * 255
        # 向内收一圈，避免边框羽化边缘残留
        valid_mask = cv2.erode(valid_mask, np.ones((7, 7), np.uint8), iterations=1)
        detection_mask = cv2.bitwise_and(detection_mask, valid_mask)

        # 2. 颜色归一化：消除整体光照差异
        normalized_photo = self.color_normalize(design, aligned_photo)

        # 2b. 轻微高斯模糊：对齐后仍有亚像素错位，会在每条边缘产生"光晕"假差异。
        #     小核模糊可显著抑制这类边缘噪声，但不影响实心色块的缺印/改色检测。
        design_cmp = cv2.GaussianBlur(design, (3, 3), 0)
        photo_cmp = cv2.GaussianBlur(normalized_photo, (3, 3), 0)

        # 3. 在mask区域内计算SSIM（只比对包装本体）
        # 先对mask外的区域填充为中性灰，避免影响SSIM
        design_masked = design_cmp.copy()
        normalized_masked = photo_cmp.copy()
        if detection_mask is not None:
            for c in range(3):
                design_masked[:, :, c] = np.where(detection_mask > 0, design_cmp[:, :, c], 128)
                normalized_masked[:, :, c] = np.where(detection_mask > 0, photo_cmp[:, :, c], 128)

        ssim_score, diff_map = self.compute_ssim(design_masked, normalized_masked)

        # 4. 像素差异
        pixel_diff = self.compute_pixel_diff(design_masked, normalized_masked)
        # 只统计mask内的像素差异
        if detection_mask is not None:
            masked_pixels = np.sum(detection_mask > 0)
            if masked_pixels > 0:
                gray_diff = cv2.cvtColor(pixel_diff, cv2.COLOR_RGB2GRAY)
                mean_pixel_diff = float(np.sum(gray_diff[detection_mask > 0]) / masked_pixels)
            else:
                mean_pixel_diff = float(np.mean(pixel_diff))
        else:
            mean_pixel_diff = float(np.mean(pixel_diff))

        # 5. 容配准误差的差异图（主信号）：在小邻域内取最小色差，
        #    吸收曲面实物导致的局部错位，只保留真正的缺印/改色
        color_diff = self.tolerant_color_diff(design_cmp, photo_cmp)
        if detection_mask is not None:
            color_diff = cv2.bitwise_and(color_diff, color_diff, mask=detection_mask)

        # 6. 找差异区域 - 以容配准差异图为主，在detection_mask上检测
        regions = self.find_difference_regions(diff_map, pixel_diff=pixel_diff,
                                               mask=detection_mask, color_diff=color_diff)
        # 6b. 二次核验：剔除仅因对齐错位造成的假差异（图标/文字边缘等）
        regions = self.verify_regions(design, normalized_photo, regions)
        # 6c. 剔除条码竖条：细高、低饱和(黑白)的区块几乎必为条码条纹，非真实缺陷
        regions = self._drop_barcode_bars(normalized_photo, design, regions)

        # 6. 可视化
        visuals = self.create_diff_visualization(design, normalized_photo, diff_map, regions, mask=detection_mask)

        # 7. 判定结果
        issues = []
        if ssim_score < self.ssim_threshold:
            issues.append(f"整体图案相似度较低 ({ssim_score:.2%} < {self.ssim_threshold:.0%})")
        if mean_pixel_diff > self.pixel_threshold * 2:
            issues.append(f"平均像素差异较大 ({mean_pixel_diff:.1f})")
        if len(regions) > 0:
            issues.append(f"发现 {len(regions)} 个显著差异区域")

        return {
            "ssim_score": ssim_score,
            "mean_pixel_diff": mean_pixel_diff,
            "diff_map": diff_map,
            "regions": regions,
            "visualizations": visuals,
            "issues": issues,
            "passed": len(issues) == 0,
            "package_mask": package_mask,
            "content_mask": content_mask,
        }
