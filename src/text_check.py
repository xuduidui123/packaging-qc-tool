"""
OCR文字提取与比对模块（EasyOCR版本）
提取设计稿和实物照片中的文字，进行比对
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
from difflib import SequenceMatcher


class TextChecker:
    """包装文字核对：检查实物文字与设计稿是否一致"""

    def __init__(self, languages=None, min_confidence: float = 0.45,
                 match_threshold: float = 0.85, mismatch_floor: float = 0.60,
                 ignore_numeric_min_len: int = 12, min_token_len: int = 2):
        """
        min_confidence:        低于此 OCR 置信度的识别结果视为噪声，直接丢弃
        match_threshold:       相似度≥此值判为"完全匹配"（放宽可减少 OCR 噪声导致的假差异）
        mismatch_floor:        相似度在 [floor, match) 之间判为"文字差异"，低于 floor 视为未匹配
        ignore_numeric_min_len:纯数字且长度≥此值的 token（整条条形码读数）跳过——OCR 对其极不可靠；
                               较短的数字（批次码/编号等）保留，以便检出真实差异
        min_token_len:         归一化后长度小于此值的 token（多为符号/杂讯，如 ®→'R'）跳过
        """
        # 延迟加载easyocr，避免导入时太慢
        self._ocr = None
        self.languages = languages or ["ch_sim", "en"]
        self._ocr_kwargs = {
            "lang_list": self.languages,
            "gpu": False,
        }
        self.min_confidence = min_confidence
        self.match_threshold = match_threshold
        self.mismatch_floor = mismatch_floor
        self.ignore_numeric_min_len = ignore_numeric_min_len
        self.min_token_len = min_token_len

    @property
    def ocr(self):
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(**self._ocr_kwargs)
        return self._ocr

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """
        从图像中提取文字
        返回: [{"text": str, "box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "confidence": float}, ...]
        """
        # EasyOCR 期望 RGB numpy array；输入已是RGB，三通道保持不变
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        result = self.ocr.readtext(image, detail=1)
        texts = []
        for item in result:
            if item is None or len(item) < 3:
                continue
            box, text, conf = item[0], item[1], item[2]
            text = text.strip()
            conf = float(conf)
            # 过滤噪声：低置信度、空/过短 token、条形码等纯数字长串
            if conf < self.min_confidence:
                continue
            if not self._is_meaningful_token(text):
                continue
            # box 格式: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            pts = np.array(box, dtype=np.float32)
            center = pts.mean(axis=0)
            texts.append({
                "text": text,
                "box": box,
                "confidence": conf,
                "center": tuple(center.tolist()),
            })
        return texts

    def _is_meaningful_token(self, text: str) -> bool:
        """判断一个 OCR token 是否值得纳入比对（过滤符号杂讯与条码类纯数字）"""
        key = self.normalize_compare(text)
        if len(key) < self.min_token_len:
            return False  # 单字符/符号类杂讯（如 ® 被读成 'R'）
        digits = "".join(ch for ch in key if ch.isdigit())
        # 纯数字（或几乎纯数字）且较长：多为条形码/长编号，OCR 不可靠，跳过
        if len(digits) >= self.ignore_numeric_min_len and len(digits) >= 0.8 * len(key):
            return False
        return True

    def normalize_compare(self, text: str) -> str:
        """比对用归一化：小写 + 仅保留字母数字（去空格/标点/符号），
        使'仅空格或标点不同'的 OCR 结果判为一致，减少假差异。"""
        text = self.normalize_text(text).lower()
        return "".join(ch for ch in text if ch.isalnum())

    def normalize_text(self, text: str) -> str:
        """文字归一化：统一空格、标点、大小写等"""
        import re
        # 全角转半角
        text = text.translate(str.maketrans(
            '０１２３４５６７８９　！＠＃＄％＾＆＊（）＿＋－＝［］｛｝；＇：＂，．／＜＞？',
            '0123456789 !@#$%^&*()_+-=[]{};\':\",./<>?'
        ))
        # 统一空格
        text = re.sub(r'\s+', ' ', text)
        # 去除前后空白
        text = text.strip()
        return text

    def text_similarity(self, text1: str, text2: str) -> float:
        """计算两段文字的相似度 (0-1)。
        使用比对归一化（忽略大小写/空格/标点），让仅排版差异的 OCR 结果判为一致。"""
        t1 = self.normalize_compare(text1)
        t2 = self.normalize_compare(text2)
        if t1 == t2:
            return 1.0
        if len(t1) == 0 and len(t2) == 0:
            return 1.0
        if len(t1) == 0 or len(t2) == 0:
            return 0.0
        return SequenceMatcher(None, t1, t2).ratio()

    def _containment_sim(self, needle: str, haystack: str) -> float:
        """needle（归一化）在 haystack（归一化全文）中的最佳出现相似度。
        用整体全文而非单个 token 做匹配，天然容忍两张图 OCR 分行/切词方式不同。"""
        n = len(needle)
        if n == 0:
            return 0.0
        if needle in haystack:
            return 1.0
        if len(haystack) < n:
            return SequenceMatcher(None, needle, haystack).ratio()
        best = 0.0
        # 滑动窗口（含 ±2 长度冗余以容纳插入/删除）
        for wlen in (n, n + 1, n + 2, max(1, n - 1)):
            if wlen <= 0 or wlen > len(haystack):
                continue
            for i in range(0, len(haystack) - wlen + 1):
                s = SequenceMatcher(None, needle, haystack[i:i + wlen]).ratio()
                if s > best:
                    best = s
                    if best >= 0.999:
                        return best
        return best

    def match_texts(self, design_texts: List[Dict], photo_texts: List[Dict],
                    position_tolerance: float = 0.15,
                    image_shape: Tuple[int, int] = None) -> List[Dict]:
        """
        基于内容的文字匹配（不再依赖位置一对一）。
        判定"某段文字是否存在"以"它是否出现在另一张图的整体文字中"为准，
        因此对两张图 OCR 分行/切词差异鲁棒，能大幅减少因切词不同导致的假缺失/假多出。
        返回: [{"type": "match"/"mismatch"/"missing"/"extra", "design", "photo", "similarity", "message"}, ...]
        """
        matches = []
        matched_photo_indices = set()

        d_norm = [self.normalize_compare(t["text"]) for t in design_texts]
        p_norm = [self.normalize_compare(t["text"]) for t in photo_texts]
        p_fulltext = "".join(p_norm)
        d_fulltext = "".join(d_norm)

        # ---- 设计稿每段文字：在实物中是否存在 ----
        for di, dt in enumerate(design_texts):
            nd = d_norm[di]
            # token 级最佳模糊匹配（用于配对展示与消费）
            best_sim, best_idx, best_pt = 0.0, -1, None
            for i, pt in enumerate(photo_texts):
                if i in matched_photo_indices:
                    continue
                sim = self.text_similarity(dt["text"], pt["text"])
                if sim > best_sim:
                    best_sim, best_idx, best_pt = sim, i, pt
            # 全文包含相似度（容忍切词/分行差异）
            cont_sim = self._containment_sim(nd, p_fulltext) if len(nd) >= self.min_token_len else 0.0
            present = max(best_sim, cont_sim)

            if present >= self.match_threshold:
                if best_pt is not None and best_sim >= self.mismatch_floor:
                    matched_photo_indices.add(best_idx)
                matches.append({
                    "type": "match", "design": dt, "photo": best_pt,
                    "similarity": present, "message": "",
                })
            elif best_pt is not None and best_sim >= self.mismatch_floor:
                matched_photo_indices.add(best_idx)
                matches.append({
                    "type": "mismatch", "design": dt, "photo": best_pt, "similarity": best_sim,
                    "message": f"设计稿: '{dt['text']}' → 实物: '{best_pt['text']}' (相似度{best_sim:.1%})"
                })
            else:
                matches.append({
                    "type": "missing", "design": dt, "photo": None, "similarity": 0.0,
                    "message": f"实物中缺失: '{dt['text']}'"
                })

        # ---- 实物中多出的文字：仅当其内容在设计稿整体文字中也找不到时才算多出 ----
        for i, pt in enumerate(photo_texts):
            if i in matched_photo_indices:
                continue
            npx = p_norm[i]
            best_sim = 0.0
            for dt in design_texts:
                s = self.text_similarity(pt["text"], dt["text"])
                if s > best_sim:
                    best_sim = s
            cont_sim = self._containment_sim(npx, d_fulltext) if len(npx) >= self.min_token_len else 0.0
            present = max(best_sim, cont_sim)
            if present >= self.mismatch_floor:
                continue  # 内容其实存在于设计稿中（只是切词不同），不算多出
            matches.append({
                "type": "extra", "design": None, "photo": pt, "similarity": 0.0,
                "message": f"实物中多出: '{pt['text']}'"
            })

        return matches

    def check(self, design_image: np.ndarray, aligned_photo: np.ndarray) -> Dict:
        """
        执行文字核对
        返回完整的核对结果
        """
        design_texts = self.extract_text(design_image)
        photo_texts = self.extract_text(aligned_photo)

        matches = self.match_texts(design_texts, photo_texts,
                                   image_shape=design_image.shape[:2])

        # 统计
        stats = {
            "total_design": len(design_texts),
            "total_photo": len(photo_texts),
            "matched": sum(1 for m in matches if m["type"] == "match"),
            "mismatched": sum(1 for m in matches if m["type"] == "mismatch"),
            "missing": sum(1 for m in matches if m["type"] == "missing"),
            "extra": sum(1 for m in matches if m["type"] == "extra"),
        }

        return {
            "design_texts": design_texts,
            "photo_texts": photo_texts,
            "matches": matches,
            "stats": stats,
        }

    def visualize(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """在图像上绘制文字核对结果"""
        vis = image.copy()
        h, w = vis.shape[:2]

        for m in result["matches"]:
            if m["type"] == "match":
                color = (0, 255, 0)  # 绿色
                # 内容匹配可能无对应实物 token，回退到设计稿框
                box = m["photo"]["box"] if m.get("photo") else m["design"]["box"]
            elif m["type"] == "mismatch":
                color = (255, 165, 0)  # 橙色
                box = m["photo"]["box"] if m.get("photo") else m["design"]["box"]
            elif m["type"] == "missing":
                color = (255, 0, 0)  # 红色
                box = m["design"]["box"]
            else:  # extra
                color = (255, 0, 255)  # 紫色
                box = m["photo"]["box"]

            if box is None:
                continue
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, color, 2)

            # 绘制文字标签
            if m["type"] == "mismatch":
                label = f"! {m['photo']['text']}"
            elif m["type"] == "missing":
                label = f"MISS: {m['design']['text']}"
            elif m["type"] == "extra":
                label = f"EXTRA: {m['photo']['text']}"
            else:
                label = m["photo"]["text"] if m.get("photo") else m["design"]["text"]

            # 找到标签位置
            top_left = tuple(np.array(box, dtype=np.int32).min(axis=0))
            y = max(20, top_left[1] - 5)
            x = top_left[0]
            # 限制在图像内
            x = min(x, w - 100)

            cv2.putText(vis, label[:30], (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        return vis
