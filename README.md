# 📦 包装设计稿与实物打样核对系统

上传「设计稿」与「实物打样照片」，系统自动完成图像对齐、文字核对（OCR）与图案差异检测，输出可视化差异报告，帮助在量产前发现印刷不一致（漏印、错字、偏色、元素缺失等）。

## 在线体验 / 部署

本项目可直接部署到 **Streamlit Community Cloud**：

1. 将本仓库推送到 GitHub。
2. 打开 https://share.streamlit.io ，用 GitHub 登录。
3. 选择本仓库，主文件填 `app.py`，点击 Deploy。

> 首次启动会下载 OCR 模型（约 100MB），需要等待几分钟。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

## 部署需要的文件

```
app.py                    # 主程序
src/                      # 核心算法模块
  ├── __init__.py
  ├── image_align.py      # 图像对齐
  ├── text_check.py       # 文字核对（OCR + 内容匹配）
  ├── pattern_check.py    # 图案差异检测
  └── visualization.py    # 报告与可视化
requirements.txt          # Python 依赖
packages.txt              # 系统依赖（apt）
.streamlit/config.toml    # 应用配置（可选）
README.md                 # 说明
```

## 使用要点

- 实物照片尽量**正面垂直平拍**、包装铺平充满画面、光照均匀、避免翘边与反光。
- 侧栏可调节 SSIM / 像素差异 / 文字匹配 三个阈值来控制灵敏度。
- 界面会给出**对齐质量提示**；对齐质量低时结果仅供参考，请人工复核。

## 技术栈

Streamlit · OpenCV · scikit-image（SSIM）· EasyOCR
