"""
包装设计稿与实物打样核对系统 - Streamlit Web界面
用法: streamlit run app.py
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageOps
import streamlit as st

# 放宽 PIL 的解压炸弹限制（我们会主动缩小大图），避免大照片直接报错
Image.MAX_IMAGE_PIXELS = None

# 加载时统一缩小到的最大边长（像素）。手机照片常达数千万像素，
# 直接处理会在云端有限内存下溢出/崩溃；缩小后既省内存又加速，精度足够。
MAX_LOAD_DIM = 2400

# 版本标识：用于确认当前运行的是否为最新代码。
# 修改 src/ 下模块后，Streamlit 的「Rerun」不会重载子模块，需完全重启服务；
# 若界面右下角/侧栏显示的版本与此不一致，说明服务未重启、仍在跑旧代码。
APP_VERSION = "v2.0 纯图像差异 (2026-07-11)"

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from image_align import ImageAligner, detect_edges_for_crop
from local_align import LocalAligner
from pattern_diff import PatternDiff
from visualization import create_full_report, empty_text_result
# 注：TextChecker（EasyOCR/torch，较重）仅在开启 OCR 时按需导入，
# 纯图像差异主流程无需这些依赖即可启动。


st.set_page_config(
    page_title="包装核对系统",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_image(uploaded_file) -> np.ndarray:
    """加载上传的图像为numpy数组 (RGB)，并对超大图自动缩小、按EXIF校正方向。"""
    image = Image.open(uploaded_file)
    # JPEG 可在解码阶段就近似缩小，显著降低大图的内存占用
    try:
        image.draft("RGB", (MAX_LOAD_DIM, MAX_LOAD_DIM))
    except Exception:
        pass
    # 依据 EXIF 方向自动旋正（手机照片常带旋转信息）
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    if image.mode != "RGB":
        image = image.convert("RGB")
    # 超大图缩小到最大边长以内，控制内存与耗时
    w, h = image.size
    if max(w, h) > MAX_LOAD_DIM:
        scale = MAX_LOAD_DIM / max(w, h)
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.array(image)


def main():
    st.title("📦 包装设计稿与实物打样核对系统")
    st.markdown("---")

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 参数设置")
        st.caption(f"版本 {APP_VERSION}")

        st.markdown("**核对模式**")
        st.caption("默认走**纯图像差异**：不识别文字，直接对齐两图、把对不上的地方框出来。"
                   "对艺术字/外文字符更稳，不受 OCR 误识别影响。")

        delta_e_threshold = st.slider(
            "色差灵敏度阈值 (ΔE)",
            min_value=8,
            max_value=40,
            value=18,
            step=1,
            help="感知色差（LAB ΔE）超过此值才视为差异。调低更灵敏（能抓轻微偏色），"
                 "调高更宽松（只报明显差异）。",
        )

        min_area_permille = st.slider(
            "最小差异面积 (‰)",
            min_value=0.2,
            max_value=10.0,
            value=0.8,
            step=0.2,
            help="小于此面积占比（千分之）的差异块被忽略，用于滤掉零星噪点。"
                 "调低可捕捉更小的缺印/瑕疵（如单个色块偏色）；调高只报大面积差异。",
        )

        high_sensitivity = st.checkbox(
            "高灵敏模式（结构/纹理通道）",
            value=False,
            help="在色差之外叠加结构/纹理通道，能抓到'颜色接近底色、只靠纹理区分'的淡纹差异，"
                 "把检测下限从约 ΔE25 降到约 ΔE10。代价：在花纹密集的水彩类设计上，"
                 "会对'两图都有、但实物印得略软'的细线产生更多误报。建议仅在怀疑有淡纹差异时开启，并人工复核。",
        )

        st.markdown("---")
        enable_ocr = st.checkbox(
            "叠加文字核对 (OCR，可选)",
            value=False,
            help="额外用 OCR 逐字比对文字。默认关闭——艺术字/外文常被 OCR 误读，"
                 "反而产生假差异。仅当你需要逐字核对且文字较规整时再开启。",
        )
        text_match_threshold = st.slider(
            "文字匹配相似度阈值",
            min_value=0.50,
            max_value=1.00,
            value=0.85,
            step=0.01,
            disabled=not enable_ocr,
            help="仅在开启 OCR 时生效。相似度≥此值视为完全匹配。",
        )

        st.markdown("---")
        st.info("""
        **使用说明：**
        1. 上传包装设计稿（PDF需先转为图片）
        2. 上传实物打样照片（尽量正面拍摄）
        3. 点击"开始核对"
        4. 查看图像差异高亮报告
        """)

    # 主区域：上传
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 设计稿")
        design_file = st.file_uploader(
            "上传设计稿",
            type=["png", "jpg", "jpeg", "bmp"],
            key="design",
            help="支持 PNG, JPG, JPEG, BMP 格式",
        )
        if design_file:
            design_img = load_image(design_file)
            st.image(design_img, use_container_width=True)

    with col2:
        st.subheader("📷 实物照片")
        photo_file = st.file_uploader(
            "上传实物照片",
            type=["png", "jpg", "jpeg", "bmp"],
            key="photo",
            help="建议正面垂直拍摄，避免强烈反光",
        )
        if photo_file:
            photo_img = load_image(photo_file)
            st.image(photo_img, use_container_width=True)

    # 核对按钮
    st.markdown("---")
    run_check = st.button("🚀 开始核对", type="primary", use_container_width=True)

    if run_check and design_file and photo_file:
        # 执行核对流程
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: 加载图像
            status_text.text("📥 正在加载图像...")
            design_img = load_image(design_file)
            photo_img = load_image(photo_file)
            progress_bar.progress(10)

            # Step 2: 图像对齐（全局透视 + 局部弹性）
            status_text.text("🔧 正在进行图像对齐（全局透视 + 局部弹性配准）...")
            aligner = ImageAligner(max_features=2000, good_match_percent=0.15)
            align_result = aligner.align(design_img, photo_img)
            global_photo = align_result["aligned_photo"]
            # 局部弹性对齐：压制曲面翘曲导致的局部错位，减少描边假高亮
            local_aligner = LocalAligner(grid=24, search_ratio=0.015)
            local_result = local_aligner.align(design_img, global_photo)
            aligned_photo = local_result["aligned_photo"]
            progress_bar.progress(40)

            # 评估对齐质量：核对结果的可靠性取决于对齐好坏
            inlier_ratio = align_result.get("inlier_ratio", 0.0)
            match_count = align_result.get("match_count", 0)
            if not align_result.get("success", False):
                align_quality = "failed"
            elif inlier_ratio < 0.45 or match_count < 30:
                align_quality = "low"
            else:
                align_quality = "good"

            # 显示对齐结果
            st.subheader("🔧 图像对齐结果")
            align_col1, align_col2, align_col3 = st.columns([2, 2, 3])
            with align_col1:
                st.image(design_img, caption="设计稿", use_container_width=True)
            with align_col2:
                st.image(aligned_photo, caption="对齐后的实物照片", use_container_width=True)
            with align_col3:
                st.info(f"**对齐状态:** {align_result['message']}")
                if align_quality == "failed":
                    st.error(
                        "⚠️ **对齐失败，核对结果不可靠。**\n\n"
                        "可能原因：照片与设计稿差异过大、拍摄过斜或模糊。\n"
                        "建议：正面垂直平拍、让包装铺平充满画面、光照均匀后重新上传。"
                    )
                elif align_quality == "low":
                    st.warning(
                        f"⚠️ **对齐质量偏低**（匹配点{match_count}个"
                        + (f"、内点率{inlier_ratio:.0%}" if inlier_ratio else "")
                        + "），下方差异可能包含较多**误报**。\n\n"
                        "建议：尽量正面垂直平拍、把包装铺平充满画面、避免翘边和反光，"
                        "可显著提高准确度。"
                    )
                else:
                    st.success(f"✅ 对齐质量良好（内点率{inlier_ratio:.0%}），结果可信度高。")
                if align_result.get("match_visualization") is not None:
                    st.image(align_result["match_visualization"],
                             caption="特征点匹配可视化", use_container_width=True)

            # Step 3: 文字核对（可选，默认跳过）
            if enable_ocr:
                status_text.text("🔤 正在提取和比对文字（OCR）...")
                from text_check import TextChecker  # 按需加载，避免启动即拉起 torch
                text_checker = TextChecker(match_threshold=text_match_threshold)
                text_result = text_checker.check(design_img, aligned_photo)
                text_vis = text_checker.visualize(aligned_photo.copy(), text_result)
                text_result["visualization"] = text_vis
            else:
                text_result = empty_text_result()
            progress_bar.progress(70)

            # Step 4: 图案核对（纯图像差异，主流程）
            status_text.text("🎨 正在检测图像差异并高亮...")
            pattern_checker = PatternDiff(
                delta_e_threshold=float(delta_e_threshold),
                min_area_ratio=min_area_permille / 1000.0,
                use_structure=high_sensitivity,
            )
            pattern_result = pattern_checker.check(design_img, aligned_photo)
            progress_bar.progress(90)

            # Step 5: 生成报告
            status_text.text("📊 正在生成报告...")
            full_report = create_full_report(
                design_img, aligned_photo, text_result, pattern_result, align_result
            )
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            # 显示总体结果
            st.markdown("---")
            overall = full_report["overall"]
            if overall["passed"]:
                st.success("✅ **核对通过！** 未发现显著差异")
            else:
                st.error("❌ **核对未通过！** 发现问题，请查看详情")

            if align_quality != "good":
                st.warning(
                    "ℹ️ 本次对齐质量不理想，以上结论仅供参考——"
                    "请结合下方差异图人工复核，或按提示重新拍摄后再核对。"
                )

            # 汇总图
            st.subheader("📊 核对汇总")
            st.image(full_report["summary_image"], use_container_width=True)

            # 详细结果 - 使用tab
            tab1, tab2, tab3 = st.tabs(["🔤 文字核对", "🎨 图案核对", "📋 完整报告"])

            with tab1:
                if text_result.get("skipped"):
                    st.info("🔤 文字核对（OCR）当前已关闭。本次为**纯图像差异**模式——"
                            "文字也走像素比对，差异会直接在「图案核对」标签里高亮框出。"
                            "如需逐字核对，请在左侧侧栏勾选「叠加文字核对 (OCR)」。")
                col_text1, col_text2 = st.columns([1, 1])
                with col_text1:
                    st.markdown("### 文字核对可视化")
                    if full_report["text_visualization"] is not None:
                        st.image(full_report["text_visualization"], use_container_width=True)
                    else:
                        st.info("文字可视化不可用")

                with col_text2:
                    st.markdown("### 统计")
                    stats = text_result["stats"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("完全匹配", stats["matched"])
                    c2.metric("文字差异", stats["mismatched"], delta_color="inverse")
                    c3.metric("实物缺失", stats["missing"], delta_color="inverse")
                    c4.metric("多出文字", stats["extra"], delta_color="inverse")

                    st.markdown("### 问题列表")
                    issues = [m for m in text_result["matches"] if m["type"] != "match"]
                    if issues:
                        for m in issues:
                            if m["type"] == "mismatch":
                                st.warning(m["message"])
                            elif m["type"] == "missing":
                                st.error(m["message"])
                            else:
                                st.info(m["message"])
                    else:
                        st.success("未发现文字问题")

            with tab2:
                col_pat1, col_pat2 = st.columns([1, 1])
                with col_pat1:
                    st.markdown("### 差异热图")
                    st.image(full_report["pattern_visualizations"]["heatmap"],
                             use_container_width=True)
                    st.markdown("### 红绿叠加图")
                    st.image(full_report["pattern_visualizations"]["red_green_overlay"],
                             use_container_width=True)

                with col_pat2:
                    st.markdown("### 差异高亮")
                    st.image(full_report["pattern_visualizations"]["diff_highlight"],
                             use_container_width=True)

                    st.markdown("### 统计")
                    st.metric("色差达标率", f"{pattern_result.get('match_rate', 0):.2%}")
                    st.metric("平均感知色差 ΔE", f"{pattern_result.get('mean_delta_e', 0):.1f}")
                    st.metric("差异区域数", len(pattern_result["regions"]))

                    if pattern_result["issues"]:
                        st.markdown("### 问题列表")
                        for issue in pattern_result["issues"]:
                            st.warning(issue)
                    else:
                        st.success("图案核对通过")

            with tab3:
                st.markdown("### 文字核对报告")
                st.text(full_report["text_report"])
                st.markdown("### 图案核对报告")
                st.text(full_report["pattern_report"])

        except Exception as e:
            st.error(f"❌ 核对过程中出现错误: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    elif run_check:
        st.warning("⚠️ 请同时上传设计稿和实物照片后再开始核对")

    # 页脚
    st.markdown("---")
    st.caption(f"包装设计稿与实物打样核对系统 {APP_VERSION} | 基于 OpenCV + EasyOCR + SSIM")


if __name__ == "__main__":
    main()
