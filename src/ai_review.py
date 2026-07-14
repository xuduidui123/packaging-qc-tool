"""
AI 复核层（可选）—— 算法定位，AI 判定
======================================
定位：不是替代算法去"找不同"（视觉大模型不擅长精确像素比对），
而是在算法已经对齐两图、圈出候选差异块之后，让视觉大模型对每个候选块做
**语义判定**：这是真缺陷（漏印/多印/改色/移位/文字变化）还是伪差异
（光照/曝光的黑↔灰、对齐残留描边、反光阴影、条码等）；并顺带指出算法可能漏掉的差异。

设计要点：
- BYOK：调用方在运行时传入自己的 API Key，本模块不持久化、不记录任何密钥。
- 多服务商：OpenAI（及兼容端点）/ Anthropic / Google Gemini，仅用标准 HTTP（requests）。
- 结构化输出：强制 JSON，低随机性；解析失败/网络错误一律返回 ok=False，由上层优雅降级。
- 控成本：图像下采样到较小边长再编码；只发"设计稿 + 带编号框的实物图"两张。
"""

from __future__ import annotations
import base64
import io
import json
import re
from typing import Dict, List, Optional

import numpy as np

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


# ------------------------------------------------------------------ 服务商配置
# protocol 决定用哪种 HTTP 协议：openai / anthropic / gemini。
# 绝大多数国内外厂商都提供「OpenAI 兼容」接口，因此只要 protocol=openai，
# 填对 base_url 与（带视觉能力的）模型名即可接入。
PROVIDERS = {
    "OpenAI": {
        "protocol": "openai",
        "default_model": "gpt-4o-mini",
        "default_base_url": "https://api.openai.com/v1",
        "key_hint": "sk-...",
        "note": "官方 GPT-4o 系列（需带视觉）。",
    },
    "Anthropic": {
        "protocol": "anthropic",
        "default_model": "claude-3-5-sonnet-latest",
        "default_base_url": "https://api.anthropic.com",
        "key_hint": "sk-ant-...",
        "note": "Claude 视觉模型。",
    },
    "Gemini": {
        "protocol": "gemini",
        "default_model": "gemini-1.5-flash",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "key_hint": "AIza...",
        "note": "Google Gemini，flash 档便宜。",
    },
    "智谱 GLM": {
        "protocol": "openai",
        "default_model": "glm-4v-flash",
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_hint": "智谱 API Key",
        "note": "OpenAI 兼容。需用视觉模型，如 glm-4v-flash / glm-4v-plus（模型名以官方文档为准）。",
    },
    "Kimi (Moonshot)": {
        "protocol": "openai",
        "default_model": "moonshot-v1-8k-vision-preview",
        "default_base_url": "https://api.moonshot.cn/v1",
        "key_hint": "sk-...",
        "note": "OpenAI 兼容。需用带 vision 的模型（模型名以官方文档为准）。",
    },
    "自定义 (OpenAI 兼容)": {
        "protocol": "openai",
        "default_model": "",
        "default_base_url": "",
        "key_hint": "你的 API Key",
        "note": "填入任意 OpenAI 兼容服务的 base_url 与视觉模型名"
                "（DeepSeek / 通义 / SiliconFlow / OpenRouter / 本地 Ollama 等均可）。",
    },
}


def provider_names() -> List[str]:
    return list(PROVIDERS.keys())


# ------------------------------------------------------------------ 图像编码
def encode_png_b64(img_rgb: np.ndarray, max_dim: int = 1024) -> str:
    """把 RGB numpy 图编码为下采样后的 PNG base64 字符串（控制体积/成本）。"""
    from PIL import Image
    im = Image.fromarray(np.asarray(img_rgb).astype("uint8"))
    w, h = im.size
    if max(w, h) > max_dim:
        s = max_dim / float(max(w, h))
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------ 提示词
def build_prompt(regions: List[Dict]) -> str:
    """构造给模型的文字指令（图像另行附带）。"""
    lines = []
    for i, r in enumerate(regions):
        x, y, w, h = r["bbox"]
        lines.append(f"  区域{i + 1}: 位置(x={x}, y={y}, w={w}, h={h})")
    region_text = "\n".join(lines) if lines else "  （算法未圈出候选区域）"
    return f"""你是包装印刷质检助手。我会给你两张图：
1) 设计稿（基准）；
2) 已对齐的实物打样照片，上面用红框和数字标出了**算法检测到的候选差异区域**。

算法擅长精确定位，但不懂语义。请你对每个带编号的候选区域做判定：它到底是
- real_defect（真缺陷：漏印/多印/图案或文字改变/移位/明显改色）；还是
- false_alarm（伪差异：光照或曝光导致的黑↔灰深浅、对齐残留的描边、反光/阴影、条码条纹等，非印刷缺陷）；还是
- uncertain（看不清/无法判断）。

候选区域清单（编号与照片上的数字一致）：
{region_text}

另外，请补充指出**算法可能漏掉**的明显差异（如果有）。

只输出 JSON，不要任何解释性文字或代码块围栏，严格用如下结构：
{{
  "regions": [
    {{"id": 1, "verdict": "real_defect|false_alarm|uncertain", "type": "missing|extra|color|text|shift|other", "description": "简短中文说明", "confidence": 0.0}}
  ],
  "missed": [
    {{"description": "算法漏掉的差异，简短中文", "location": "大致位置，如左上/条码上方", "bbox": [0.0, 0.0, 0.0, 0.0], "confidence": 0.0}}
  ],
  "overall": "pass|fail|review",
  "summary": "一句话中文总体结论"
}}
说明：
- confidence 为 0~1 的数字，请给出你的真实把握度（不要一律填 0）。
- missed 中的 bbox 为该漏检差异在**第二张实物照片**上的**归一化**位置 [x, y, w, h]，
  取值 0~1（x,y 为左上角，w,h 为宽高，均相对整幅图的比例）。请务必尽量给出，用于在图上标注箭头。
- 若没有漏检项，missed 用空数组 []。"""


# ------------------------------------------------------------------ JSON 解析
def _first_json_object(s: str) -> Optional[str]:
    """用括号配平扫描出第一个完整的 {...} 对象（忽略字符串内的花括号）。"""
    start = s.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None


def _repair_json(s: str) -> str:
    """轻度修复：去掉对象/数组结尾的多余逗号。"""
    return re.sub(r",\s*([}\]])", r"\1", s)


def _extract_json(text: str) -> Optional[dict]:
    """稳健解析模型返回：去思维链/代码围栏 → 括号配平提取 → 轻度修复。"""
    if not text:
        return None
    t = text.strip()
    # 去掉部分模型的思维链块
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    # 若包裹在 ``` 代码围栏里，取围栏内内容
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.S)
    if fence:
        t = fence.group(1).strip()

    candidates = [t]
    obj = _first_json_object(t)
    if obj and obj != t:
        candidates.append(obj)

    for cand in candidates:
        for variant in (cand, _repair_json(cand)):
            try:
                data = json.loads(variant)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return None


# ------------------------------------------------------------------ 各服务商调用
def _call_openai(design_b64, highlight_b64, prompt, api_key, model, base_url, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 2000,  # 防止 JSON 被截断
        "response_format": {"type": "json_object"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "第一张：设计稿"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{design_b64}"}},
                {"type": "text", "text": "第二张：带编号候选框的实物照片"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{highlight_b64}"}},
            ],
        }],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    try:
        resp.raise_for_status()
    except Exception:
        # 部分 OpenAI 兼容端点不支持 response_format，去掉后重试一次
        if "response_format" in body:
            body.pop("response_format", None)
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
        else:
            raise
    return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(design_b64, highlight_b64, prompt, api_key, model, base_url, timeout):
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    body = {
        "model": model,
        "max_tokens": 1500,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "第一张：设计稿"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": design_b64}},
                {"type": "text", "text": "第二张：带编号候选框的实物照片"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": highlight_b64}},
            ],
        }],
    }
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return "".join(blk.get("text", "") for blk in data.get("content", []))


def _call_gemini(design_b64, highlight_b64, prompt, api_key, model, base_url, timeout):
    url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        "contents": [{
            "parts": [
                {"text": prompt},
                {"text": "第一张：设计稿"},
                {"inline_data": {"mime_type": "image/png", "data": design_b64}},
                {"text": "第二张：带编号候选框的实物照片"},
                {"inline_data": {"mime_type": "image/png", "data": highlight_b64}},
            ],
        }],
    }
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# 按协议分派（openai 协议覆盖 OpenAI / GLM / Kimi / 自定义等兼容端点）
_DISPATCH = {"openai": _call_openai, "anthropic": _call_anthropic, "gemini": _call_gemini}


# ------------------------------------------------------------------ 主入口
def review(design_img: np.ndarray,
           highlight_img: np.ndarray,
           regions: List[Dict],
           provider: str,
           api_key: str,
           model: Optional[str] = None,
           base_url: Optional[str] = None,
           max_dim: int = 1024,
           timeout: int = 60) -> Dict:
    """
    调用视觉大模型对算法候选区域做语义复核。
    返回：{"ok": bool, "data": {...} | None, "error": str | None, "raw": str | None}
    data 结构见 build_prompt 中的 JSON schema。任何异常都被捕获为 ok=False，便于上层降级。
    """
    if requests is None:
        return {"ok": False, "error": "缺少 requests 库，无法调用 API。", "data": None, "raw": None}
    if provider not in PROVIDERS:
        return {"ok": False, "error": f"不支持的服务商: {provider}", "data": None, "raw": None}
    if not api_key or not api_key.strip():
        return {"ok": False, "error": "未提供 API Key。", "data": None, "raw": None}

    cfg = PROVIDERS[provider]
    protocol = cfg["protocol"]
    model = (model or cfg["default_model"] or "").strip()
    base_url = (base_url or cfg["default_base_url"] or "").strip()
    if not base_url:
        return {"ok": False, "error": "未填写接口地址 (base_url)。", "data": None, "raw": None}
    if not model:
        return {"ok": False, "error": "未填写模型名称。", "data": None, "raw": None}

    try:
        design_b64 = encode_png_b64(design_img, max_dim=max_dim)
        highlight_b64 = encode_png_b64(highlight_img, max_dim=max_dim)
        prompt = build_prompt(regions)
        raw = _DISPATCH[protocol](design_b64, highlight_b64, prompt,
                                  api_key.strip(), model, base_url, timeout)
    except Exception as e:  # 网络/鉴权/额度等一律降级
        msg = str(e)
        # 避免把可能包含密钥的内容回显
        if api_key and api_key in msg:
            msg = msg.replace(api_key, "***")
        return {"ok": False, "error": f"调用失败：{msg}", "data": None, "raw": None}

    data = _extract_json(raw)
    if data is None:
        return {"ok": False, "error": "模型返回无法解析为 JSON。", "data": None, "raw": raw}
    # 规整字段，缺失给默认值
    data.setdefault("regions", [])
    data.setdefault("missed", [])
    data.setdefault("overall", "review")
    data.setdefault("summary", "")
    return {"ok": True, "data": data, "error": None, "raw": raw}
