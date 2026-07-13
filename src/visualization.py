"""
可视化报告生成
整合文字和图案核对结果，生成完整的可视化报告
"""

import numpy as np
import cv2
from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont


def create_summary_image(design: np.ndarray, aligned_photo: np.ndarray,
                         text_result: Dict, pattern_result: Dict) -> np.ndarray:
    """
    创建汇总对比图：设计稿 | 实物 | 文字差异 | 图案差异
    """
    h, w = design.shape[:2]

    # 缩放以保持合理尺寸
    max_height = 600
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = max_height
        design = cv2.resize(design, (new_w, new_h))
        aligned_photo = cv2.resize(aligned_photo, (new_w, new_h))
        h, w = new_h, new_w
    else:
        new_w, new_h = w, h

    # 获取可视化图
    text_vis = text_result.get("visualization")
    if text_vis is not None:
        text_vis = cv2.resize(text_vis, (new_w, new_h))
    else:
        text_vis = aligned_photo.copy()

    diff_highlight = pattern_result["visualizations"]["diff_highlight"]
    diff_highlight = cv2.resize(diff_highlight, (new_w, new_h))

    # 标签行
    label_h = 30
    labels = ["设计稿", "实物照片", "文字核对", "图案核对"]

    # 创建画布
    canvas_w = new_w * 4
    canvas_h = new_h + label_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # 放置图像
    canvas[label_h:label_h+new_h, 0:new_w] = design
    canvas[label_h:label_h+new_h, new_w:2*new_w] = aligned_photo
    canvas[label_h:label_h+new_h, 2*new_w:3*new_w] = text_vis
    canvas[label_h:label_h+new_h, 3*new_w:4*new_w] = diff_highlight

    # 添加标签（用OpenCV）
    for i, label in enumerate(labels):
        x = i * new_w + 10
        y = 22
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

    return canvas


def create_text_report(text_result: Dict) -> str:
    """生成文字核对文字报告"""
    stats = text_result["stats"]
    lines = []
    lines.append("=" * 50)
    lines.append("📋 文字核对报告")
    lines.append("=" * 50)
    lines.append(f"设计稿文字数: {stats['total_design']}")
    lines.append(f"实物文字数:   {stats['total_photo']}")
    lines.append(f"✅ 完全匹配:   {stats['matched']}")
    lines.append(f"⚠️  文字差异:   {stats['mismatched']}")
    lines.append(f"❌ 实物缺失:   {stats['missing']}")
    lines.append(f"🔍 多出文字:   {stats['extra']}")
    lines.append("")

    if stats["mismatched"] > 0:
        lines.append("--- 文字差异详情 ---")
        for m in text_result["matches"]:
            if m["type"] == "mismatch":
                lines.append(m["message"])
        lines.append("")

    if stats["missing"] > 0:
        lines.append("--- 缺失文字详情 ---")
        for m in text_result["matches"]:
            if m["type"] == "missing":
                lines.append(m["message"])
        lines.append("")

    if stats["extra"] > 0:
        lines.append("--- 多出文字详情 ---")
        for m in text_result["matches"]:
            if m["type"] == "extra":
                lines.append(m["message"])
        lines.append("")

    return "\n".join(lines)


def create_pattern_report(pattern_result: Dict) -> str:
    """生成图案核对文字报告"""
    lines = []
    lines.append("=" * 50)
    lines.append("🎨 图像差异核对报告")
    lines.append("=" * 50)
    # 兼容新旧两套指标：新流程用 match_rate/mean_delta_e，旧流程用 ssim/pixel
    if "match_rate" in pattern_result:
        lines.append(f"色差达标率:   {pattern_result['match_rate']:.2%}")
        lines.append(f"平均感知色差: {pattern_result.get('mean_delta_e', 0):.1f}")
    else:
        lines.append(f"SSIM结构相似度: {pattern_result.get('ssim_score', 0):.2%}")
        lines.append(f"平均像素差异:   {pattern_result.get('mean_pixel_diff', 0):.1f}")
    lines.append(f"差异区域数:     {len(pattern_result['regions'])}")
    lines.append("")

    if pattern_result["passed"]:
        lines.append("✅ 图案核对通过，未发现显著差异")
    else:
        lines.append("❌ 发现图案差异:")
        for issue in pattern_result["issues"]:
            lines.append(f"  • {issue}")

    if pattern_result["regions"]:
        lines.append("")
        lines.append("--- 差异区域详情 ---")
        for i, r in enumerate(pattern_result["regions"], 1):
            x, y, w, h = r["bbox"]
            lines.append(f"  区域{i}: 位置({x},{y}) 大小{w}x{h} 差异强度{r['avg_diff']:.1f}")

    return "\n".join(lines)


def empty_text_result() -> Dict:
    """OCR 关闭时的占位文字结果，让报告层无需改动即可跳过文字核对。"""
    return {
        "stats": {"total_design": 0, "total_photo": 0, "matched": 0,
                  "mismatched": 0, "missing": 0, "extra": 0},
        "matches": [],
        "visualization": None,
        "skipped": True,
    }


def create_full_report(design: np.ndarray, aligned_photo: np.ndarray,
                       text_result: Dict, pattern_result: Dict,
                       align_info: Dict) -> Dict:
    """
    生成完整的核对报告
    返回包含所有可视化图和文字报告的字典
    """
    summary_img = create_summary_image(design, aligned_photo, text_result, pattern_result)
    text_report = create_text_report(text_result)
    pattern_report = create_pattern_report(pattern_result)

    # 整体判定
    text_pass = (text_result["stats"]["mismatched"] == 0 and
                 text_result["stats"]["missing"] == 0 and
                 text_result["stats"]["extra"] == 0)
    pattern_pass = pattern_result["passed"]

    overall = {
        "passed": text_pass and pattern_pass,
        "text_pass": text_pass,
        "pattern_pass": pattern_pass,
        "align_success": align_info.get("success", False),
        "align_message": align_info.get("message", ""),
    }

    return {
        "summary_image": summary_img,
        "text_report": text_report,
        "pattern_report": pattern_report,
        "text_visualization": text_result.get("visualization"),
        "pattern_visualizations": pattern_result["visualizations"],
        "overall": overall,
    }
