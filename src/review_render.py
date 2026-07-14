"""
AI 复核批注视图渲染
==================
把「带编号框的图」和「AI 逐区判定 / 漏检项」用**箭头连线**关联起来：
左边是实物照片（算法候选框按 AI 判定着色 + 编号；AI 指出的漏检项用虚线橙框标注），
右边是对应的判定卡片，两者用正交连接线一一相连。

产出纯 SVG（内嵌 base64 图片），由 app 通过 components.html 渲染。
无第三方依赖（PIL 用于把 numpy 图编码为 PNG）。
"""

from __future__ import annotations
import base64
import io
import re
from html import escape
from typing import Dict, List, Optional

import numpy as np

# 服务端栅格化(cairosvg)时使用的字体：浏览器用 PingFang/雅黑，Linux 服务器用 Noto CJK
RASTER_FONT = "Noto Sans CJK SC,Noto Serif CJK SC,WenQuanYi Zen Hei,sans-serif"

# 判定 -> (中文名, 颜色)
VERDICT_STYLE = {
    "real_defect": ("真缺陷", "#D64545"),
    "false_alarm": ("伪差异", "#5B7FA6"),
    "uncertain": ("不确定", "#D89614"),
}
MISSED_COLOR = "#E8720C"
FONT = "-apple-system,'PingFang SC','Microsoft YaHei',sans-serif"


def _png_b64(img_rgb: np.ndarray, max_dim: int = 1000) -> str:
    from PIL import Image
    im = Image.fromarray(np.asarray(img_rgb).astype("uint8"))
    w, h = im.size
    if max(w, h) > max_dim:
        s = max_dim / float(max(w, h))
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _wrap(text: str, n: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    return [text[i:i + n] for i in range(0, len(text), n)] or [""]


def _conf_str(c) -> str:
    try:
        c = float(c)
    except Exception:
        return ""
    return f"（把握度 {c:.0%}）" if c > 0 else ""


def build_review_svg(base_rgb: np.ndarray,
                     regions: List[Dict],
                     ai_data: Dict,
                     disp_w: int = 700) -> str:
    """
    生成批注视图的**裸 SVG 字符串**。
    base_rgb : 用作底图的实物照片（对齐后，RGB）
    regions  : 算法差异块列表（像素 bbox，(x,y,w,h)），顺序与图上编号 1..N 对应
    ai_data  : AI 复核 JSON（regions: [{id,verdict,type,description,confidence}], missed: [...]）
    """
    img_h, img_w = base_rgb.shape[:2]
    scale = disp_w / float(img_w)
    disp_h = img_h * scale
    pad = 16
    gap = 92                      # 连线通道宽度
    card_w = 400
    x_img = pad
    x_col = pad + disp_w + gap    # 右侧卡片列左边缘
    svg_w = x_col + card_w + pad

    ai_regions = {int(r.get("id", -1)): r for r in ai_data.get("regions", []) if str(r.get("id", "")).lstrip("-").isdigit()}
    missed = ai_data.get("missed", []) or []

    # 组装带定位的条目（有 target 的画连线）
    items = []
    for idx, reg in enumerate(regions):
        rid = idx + 1
        x, y, w, h = reg["bbox"]
        cx, cy = (x + w / 2) * scale + x_img, (y + h / 2) * scale + pad
        av = ai_regions.get(rid, {})
        verdict = str(av.get("verdict", "uncertain"))
        name, color = VERDICT_STYLE.get(verdict, VERDICT_STYLE["uncertain"])
        typ = av.get("type", "")
        desc = av.get("description", "") or "（无 AI 说明）"
        items.append({
            "badge": str(rid), "color": color, "dashed": False,
            "box": (x * scale + x_img, y * scale + pad, w * scale, h * scale),
            "target": (cx, cy),
            "header": f"区域 {rid} · {name}" + (f" · {typ}" if typ else ""),
            "desc": desc, "conf": av.get("confidence"),
        })
    for j, m in enumerate(missed):
        bb = m.get("bbox")
        box = target = None
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            try:
                nx, ny, nw, nh = [float(v) for v in bb]
                if nw > 0 and nh > 0:
                    box = (nx * img_w * scale + x_img, ny * img_h * scale + pad,
                           nw * img_w * scale, nh * img_h * scale)
                    target = (box[0] + box[2] / 2, box[1] + box[3] / 2)
            except Exception:
                pass
        loc = m.get("location", "")
        items.append({
            "badge": f"M{j + 1}", "color": MISSED_COLOR, "dashed": True,
            "box": box, "target": target,
            "header": f"漏检 M{j + 1}" + (f" · {loc}" if loc else ""),
            "desc": m.get("description", "") or "（无说明）", "conf": m.get("confidence"),
        })

    # 卡片高度估算（按换行行数）
    def card_h(it):
        lines = _wrap(it["desc"], 22)
        return 30 + 20 + len(lines) * 19 + 14  # header + gap + desc lines + padding

    # 有 target 的按目标 y 排序，减少连线交叉；无 target 的放最后
    with_t = sorted([it for it in items if it["target"]], key=lambda it: it["target"][1])
    without_t = [it for it in items if not it["target"]]
    ordered = with_t + without_t

    # 竖直堆叠卡片
    y_cursor = pad
    for it in ordered:
        ch = card_h(it)
        it["card_y"] = y_cursor
        it["card_h"] = ch
        y_cursor += ch + 12

    cards_bottom = y_cursor
    svg_h = int(max(disp_h + 2 * pad, cards_bottom + pad))

    img_b64 = _png_b64(base_rgb)
    mx = pad + disp_w + gap * 0.5   # 连线中段折点 x

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" '
        f'style="max-width:100%;height:auto;font-family:{FONT}">',
        f'<rect x="0" y="0" width="{svg_w}" height="{svg_h}" fill="#ffffff"/>',
        f'<image x="{x_img}" y="{pad}" width="{disp_w}" height="{disp_h:.1f}" '
        f'href="data:image/png;base64,{img_b64}" preserveAspectRatio="none"/>',
        f'<rect x="{x_img}" y="{pad}" width="{disp_w}" height="{disp_h:.1f}" '
        f'fill="none" stroke="#e0e0e0"/>',
    ]

    # 连线（先画，压在框/卡下层）
    for it in ordered:
        if not it["target"]:
            continue
        tx, ty = it["target"]
        cyc = it["card_y"] + 22
        parts.append(
            f'<polyline points="{tx:.0f},{ty:.0f} {mx:.0f},{ty:.0f} {mx:.0f},{cyc:.0f} {x_col:.0f},{cyc:.0f}" '
            f'fill="none" stroke="{it["color"]}" stroke-width="1.6" opacity="0.85"/>'
        )
        parts.append(f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="3.5" fill="{it["color"]}"/>')

    # 图上的框 + 编号徽标
    for it in ordered:
        if it["box"]:
            bx, by, bw, bh = it["box"]
            dash = 'stroke-dasharray="6 4" ' if it["dashed"] else ""
            parts.append(
                f'<rect x="{bx:.0f}" y="{by:.0f}" width="{bw:.0f}" height="{bh:.0f}" '
                f'fill="{it["color"]}" fill-opacity="0.12" stroke="{it["color"]}" stroke-width="2.4" {dash}/>'
            )
            parts.append(
                f'<circle cx="{bx:.0f}" cy="{by:.0f}" r="11" fill="{it["color"]}"/>'
                f'<text x="{bx:.0f}" y="{by + 4:.0f}" text-anchor="middle" font-size="13" '
                f'fill="#fff" font-weight="700">{escape(it["badge"])}</text>'
            )

    # 右侧卡片
    for it in ordered:
        cy0 = it["card_y"]
        ch = it["card_h"]
        parts.append(
            f'<rect x="{x_col}" y="{cy0}" width="{card_w}" height="{ch}" rx="8" '
            f'fill="#fafafa" stroke="#e6e6e6"/>'
            f'<rect x="{x_col}" y="{cy0}" width="5" height="{ch}" rx="2" fill="{it["color"]}"/>'
        )
        # 徽标
        parts.append(
            f'<circle cx="{x_col + 24}" cy="{cy0 + 24}" r="12" fill="{it["color"]}"/>'
            f'<text x="{x_col + 24}" y="{cy0 + 28}" text-anchor="middle" font-size="12" '
            f'fill="#fff" font-weight="700">{escape(it["badge"])}</text>'
        )
        conf = _conf_str(it["conf"])
        parts.append(
            f'<text x="{x_col + 44}" y="{cy0 + 22}" font-size="14.5" font-weight="700" '
            f'fill="#222">{escape(it["header"])}</text>'
        )
        if conf:
            parts.append(
                f'<text x="{x_col + 44}" y="{cy0 + 40}" font-size="12" fill="#999">{escape(conf)}</text>'
            )
        # 说明换行
        y_text = cy0 + (58 if conf else 44)
        for line in _wrap(it["desc"], 22):
            parts.append(
                f'<text x="{x_col + 16}" y="{y_text:.0f}" font-size="13.5" fill="#444">{escape(line)}</text>'
            )
            y_text += 19

    parts.append("</svg>")
    return "".join(parts)


def build_review_html(base_rgb: np.ndarray, regions: List[Dict], ai_data: Dict,
                      disp_w: int = 700) -> str:
    """浏览器内嵌用：裸 SVG 外包一层可滚动 div。"""
    svg = build_review_svg(base_rgb, regions, ai_data, disp_w=disp_w)
    return f'<div style="overflow:auto">{svg}</div>'


def svg_to_png(svg: str, output_width: int = 1400) -> bytes:
    """把批注 SVG 栅格化为 PNG 字节（服务端，用 cairosvg）。
    需要 cairosvg 及中文字体（部署时见 packages.txt: libcairo2 / fonts-noto-cjk）。"""
    import cairosvg
    # 换成服务器上存在的 CJK 字体；去掉根 svg 的 style（height:auto 会让 cairosvg 算成 0）
    s = svg.replace(FONT, RASTER_FONT)
    s = re.sub(r'(<svg[^>]*?)\sstyle="[^"]*"', r"\1", s, count=1)
    return cairosvg.svg2png(bytestring=s.encode("utf-8"),
                            output_width=output_width, background_color="white")


def estimate_height(base_rgb: np.ndarray, regions: List[Dict], ai_data: Dict,
                    disp_w: int = 700) -> int:
    """粗略估算渲染高度（供 components.html 设定容器高度）。"""
    img_h, img_w = base_rgb.shape[:2]
    disp_h = img_h * (disp_w / float(img_w))
    n_regions = len(regions)
    n_missed = len(ai_data.get("missed", []) or [])
    approx_card = 92
    cards_total = (n_regions + n_missed) * approx_card + 32
    return int(max(disp_h + 32, cards_total)) + 24
