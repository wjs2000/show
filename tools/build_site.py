# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import html
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets"
PROJECTS_DIR = REPO / "projects"

SOURCE_ROOT = Path(r"G:\云南财经大学\工作实习\项目文档")
DECISION_ROOT = Path(r"G:\云南财经大学\决策支持系统\量化系统v1")


def ensure_clean() -> None:
    def clear_readonly(func, path, exc_info):
        try:
            os.chmod(path, 0o700)
            func(path)
        except Exception:
            raise exc_info[1]

    for path in [ASSETS / "images", ASSETS / "media", PROJECTS_DIR]:
        if path.exists():
            shutil.rmtree(path, onerror=clear_readonly)
        path.mkdir(parents=True, exist_ok=True)


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def slug_path(*parts: str) -> Path:
    path = REPO
    for part in parts:
        path = path / part
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path, from_dir: Path | None = None) -> str:
    if from_dir is None:
        from_dir = REPO
    return path.relative_to(from_dir).as_posix()


def find_file(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return p


def copy_image(src: Path, dest_rel: str, max_w: int = 1400, max_h: int = 900, quality: int = 86) -> Path:
    dest = slug_path(dest_rel)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        if dest.suffix.lower() in [".jpg", ".jpeg", ".webp"]:
            if im.mode == "RGBA":
                bg = Image.new("RGB", im.size, "white")
                bg.paste(im, mask=im.split()[-1])
                im = bg
        if dest.suffix.lower() == ".webp":
            im.save(dest, "WEBP", quality=quality, method=6)
        elif dest.suffix.lower() in [".jpg", ".jpeg"]:
            im.save(dest, "JPEG", quality=quality, optimize=True, progressive=True)
        else:
            im.save(dest, optimize=True)
    return dest


def copy_static(src: Path, dest_rel: str) -> Path:
    dest = slug_path(dest_rel)
    shutil.copy2(src, dest)
    return dest


def make_video_gif(
    src: Path,
    dest_rel: str,
    start_sec: float = 0,
    duration: float = 3.2,
    fps_out: int = 8,
    width: int = 420,
    max_frames: int = 32,
) -> Path:
    dest = slug_path(dest_rel)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = int(start_sec * fps_in)
    end_frame = min(total, start_frame + int(duration * fps_in)) if total else start_frame + int(duration * fps_in)
    step = max(1, int(fps_in / fps_out))
    frames: list[Image.Image] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while idx < end_frame and len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start_frame) % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            new_w = width
            new_h = max(1, int(h * new_w / w))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            im = Image.fromarray(frame)
            frames.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted: {src}")
    frames[0].save(
        dest,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps_out),
        loop=0,
        optimize=True,
    )
    return dest


def make_reveal_gif(src: Path, dest_rel: str, width: int = 680, frames_count: int = 30) -> Path:
    dest = slug_path(dest_rel)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        ratio = width / im.width
        height = max(1, int(im.height * ratio))
        im = im.resize((width, height), Image.Resampling.LANCZOS)
    frames: list[Image.Image] = []
    bg = Image.new("RGB", im.size, (238, 241, 244))
    draw = ImageDraw.Draw(bg)
    step_lines = max(4, width // 16)
    for x in range(0, width, step_lines):
        draw.line([(x, 0), (x, height)], fill=(220, 225, 230))
    for i in range(frames_count):
        t = (i + 1) / frames_count
        reveal_w = int(width * t)
        frame = bg.copy()
        crop = im.crop((0, 0, reveal_w, height))
        frame.paste(crop, (0, 0))
        d = ImageDraw.Draw(frame)
        d.line([(reveal_w, 0), (reveal_w, height)], fill=(32, 120, 160), width=3)
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    frames.extend([frames[-1]] * 6)
    frames[0].save(dest, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    return dest


def draw_quant_charts(eval_csv: Path, top_csv: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False

    eval_rows = []
    with eval_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "test":
                eval_rows.append(row)
    models = [r["model"] for r in eval_rows]
    auc = [float(r["auc"]) for r in eval_rows]
    p3 = [float(r["precision_at_3"]) for r in eval_rows]

    chart1 = slug_path("assets/images/quant/model-evaluation-zoom.png")
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=170)
    x = np.arange(len(models))
    ax.bar(x - 0.18, auc, 0.34, label="Test AUC", color="#2b7a78")
    ax.bar(x + 0.18, p3, 0.34, label="P@3", color="#b55d2a")
    ax.set_xticks(x, models)
    ymin = max(0.48, min(auc + p3) - 0.02)
    ymax = min(0.62, max(auc + p3) + 0.03)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Model Evaluation on Test Split")
    for xpos, value in zip(list(x - 0.18) + list(x + 0.18), auc + p3):
        ax.text(xpos, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(chart1, bbox_inches="tight")
    plt.close(fig)

    top_rows = []
    with top_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") == "Logistic":
                top_rows.append(row)
    top_rows = sorted(top_rows, key=lambda r: int(float(r["rank_num"])))[:3]
    names = [r["etf_name"] for r in top_rows]
    probs = [float(r["pred_probability"]) for r in top_rows]
    chart2 = slug_path("assets/images/quant/latest-top3-zoom.png")
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=170)
    ypos = np.arange(len(names))
    ax.barh(ypos, probs, color=["#256d85", "#5c946e", "#c47f33"])
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlim(max(0.50, min(probs) - 0.025), min(0.62, max(probs) + 0.035))
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Latest Logistic Top 3 Probability")
    for y, value in zip(ypos, probs):
        ax.text(value + 0.002, y, f"{value:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(chart2, bbox_inches="tight")
    plt.close(fig)
    return chart1, chart2


def make_miniapp_collage(srcs: list[Path], dest_rel: str) -> Path:
    dest = slug_path(dest_rel)
    thumbs = []
    for src in srcs:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((260, 520), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (280, 560), (246, 247, 249))
            x = (canvas.width - im.width) // 2
            y = (canvas.height - im.height) // 2
            canvas.paste(im, (x, y))
            thumbs.append(canvas)
    w = 280 * len(thumbs)
    h = 560
    out = Image.new("RGB", (w, h), (235, 238, 241))
    for i, im in enumerate(thumbs):
        out.paste(im, (i * 280, 0))
    out.thumbnail((1200, 720), Image.Resampling.LANCZOS)
    out.save(dest, "JPEG", quality=88, optimize=True, progressive=True)
    return dest


def media_tag(src: str, alt: str = "", cls: str = "", caption: str | None = None, poster: str | None = None) -> str:
    ext = Path(src).suffix.lower()
    if ext == ".mp4":
        poster_attr = f' poster="{esc(poster)}"' if poster else ""
        body = f'<video class="{cls}" controls muted playsinline preload="metadata"{poster_attr}><source src="{esc(src)}" type="video/mp4"></video>'
    else:
        body = f'<img class="{cls}" src="{esc(src)}" alt="{esc(alt)}" loading="lazy">'
    if caption:
        return f'<figure>{body}<figcaption>{esc(caption)}</figcaption></figure>'
    return body


def image_card(src: str, title: str, desc: str = "") -> str:
    desc_html = f"<p>{esc(desc)}</p>" if desc else ""
    return f"""
    <article class="media-card">
      {media_tag(src, title)}
      <div>
        <h4>{esc(title)}</h4>
        {desc_html}
      </div>
    </article>
    """


def metric_cards(metrics: Iterable[tuple[str, str]]) -> str:
    return '<div class="metrics">' + "".join(
        f'<div class="metric"><span>{esc(k)}</span><strong>{esc(v)}</strong></div>' for k, v in metrics
    ) + "</div>"


def carousel(images: list[tuple[str, str]], extra_class: str = "") -> str:
    slides = "".join(
        f'<figure class="slide {"is-active" if i == 0 else ""}">{media_tag(src, alt)}<figcaption>{esc(alt)}</figcaption></figure>'
        for i, (src, alt) in enumerate(images)
    )
    dots = "".join(f'<button aria-label="切换到第{i+1}张" class="{"is-active" if i == 0 else ""}"></button>' for i in range(len(images)))
    return f'<div class="auto-carousel {extra_class}" data-carousel>{slides}<div class="carousel-dots">{dots}</div></div>'


def page_shell(title: str, subtitle: str, body: str, active: str = "") -> str:
    nav_items = [
        ("../index.html" if active else "index.html", "作品集首页"),
        ("avm-360.html", "360环视"),
        ("acaf-finance.html", "AFAC金融"),
        ("ai-fitness.html", "AI健身"),
        ("quant-agent.html", "量化Agent"),
        ("railway-stitching.html", "铁路拼接"),
        ("logistics-llm.html", "物流大模型"),
        ("miniapps.html", "小程序合集"),
    ]
    if not active:
        nav_items = [(href if href == "index.html" else f"projects/{href}", text) for href, text in nav_items]
    nav = "".join(f'<a href="{esc(href)}">{esc(text)}</a>' for href, text in nav_items)
    prefix = "../" if active else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)} | 王俊松项目作品集</title>
  <meta name="description" content="{esc(subtitle)}">
  <link rel="stylesheet" href="{prefix}assets/styles.css">
</head>
<body>
  <header class="site-header">
    <a class="brand" href="{prefix}index.html">王俊松 · 项目作品集</a>
    <nav>{nav}</nav>
  </header>
  <main>
    {body}
  </main>
  <footer class="site-footer">
    <span>Portfolio for internship interviews</span>
    <span>Computer Vision · Financial AI · RAG · Mini Programs</span>
  </footer>
  <script src="{prefix}assets/site.js"></script>
</body>
</html>
"""


def hero(title: str, kicker: str, desc: str, visual_html: str, tags: list[str]) -> str:
    tags_html = "".join(f"<span>{esc(t)}</span>" for t in tags)
    return f"""
    <section class="project-hero">
      <div class="hero-copy">
        <p class="kicker">{esc(kicker)}</p>
        <h1>{esc(title)}</h1>
        <p class="lead">{esc(desc)}</p>
        <div class="tag-row">{tags_html}</div>
      </div>
      <div class="hero-visual">{visual_html}</div>
    </section>
    """


def section(title: str, content: str, cls: str = "") -> str:
    return f'<section class="section {cls}"><div class="section-title"><h2>{esc(title)}</h2></div>{content}</section>'


def write_assets() -> dict[str, str]:
    paths: dict[str, str] = {}

    avm_base = SOURCE_ROOT / "360车载全景影像拼接" / "portfolio_pack"
    avm_demo = avm_base / "素材展示" / "上机演示图片"
    avm_show = avm_base / "showcase_materials" / "resume_zip_showcase" / "images"
    for name, key in [
        ("上机前.jpg", "avm_demo_front"),
        ("上机右.jpg", "avm_demo_right"),
        ("上机后.jpg", "avm_demo_back"),
        ("上机左.jpg", "avm_demo_left"),
        ("上机鸟瞰图.jpg", "avm_demo_bev"),
    ]:
        paths[key] = rel(copy_image(avm_demo / name, f"assets/images/avm/{key}.jpg", 1200, 780))
    for name, key in [
        ("05_baseline_bev.png", "avm_baseline"),
        ("06_owned_regions_best.png", "avm_owned"),
        ("07_extended_safe.png", "avm_safe"),
        ("08_extended_wide.png", "avm_wide"),
    ]:
        paths[key] = rel(copy_image(avm_show / name, f"assets/images/avm/{key}.jpg", 1300, 900))

    acaf_img = SOURCE_ROOT / "ACAF2025金融智能创新大赛" / "portfolio_pack" / "assets" / "github_pages_materials" / "images"
    for name, key in [
        ("路演现场.jpg", "acaf_roadshow"),
        ("颁奖仪式.jpg", "acaf_award"),
        ("奖金奖杯.jpg", "acaf_trophy"),
        ("证书.jpg", "acaf_certificate"),
    ]:
        paths[key] = rel(copy_image(acaf_img / name, f"assets/images/acaf/{key}.jpg", 1200, 760))
    acaf_charts = SOURCE_ROOT / "ACAF2025金融智能创新大赛" / "portfolio_pack" / "assets" / "charts"
    for name, key in [
        ("prediction_7day_aggregate.png", "acaf_pred"),
        ("fund_prediction_top10.png", "acaf_top10"),
        ("training_monthly_actuals.png", "acaf_monthly"),
        ("feature_correlation_snapshot.png", "acaf_corr"),
    ]:
        paths[key] = rel(copy_image(acaf_charts / name, f"assets/images/acaf/{key}.png", 1100, 760))
    paths["acaf_arch"] = rel(copy_static(acaf_charts / "architecture_dataflow.svg", "assets/images/acaf/architecture.svg"))

    fitness = SOURCE_ROOT / "AI健身项目" / "portfolio_pack" / "素材包"
    paths["fitness_home"] = rel(copy_image(fitness / "首页图" / "首页图.png", "assets/images/fitness/home.jpg", 1200, 760))
    for name, key in [
        ("深蹲-展示.png", "fitness_squat"),
        ("俯卧撑-展示.png", "fitness_pushup"),
        ("引体向上-展示.png", "fitness_pullup"),
        ("仰卧起坐-展示.png", "fitness_situp"),
    ]:
        paths[key] = rel(copy_image(fitness / "截图" / name, f"assets/images/fitness/{key}.jpg", 760, 980))
    paths["fitness_gif_squat"] = rel(make_video_gif(fitness / "视频" / "out-深蹲-左侧面.mp4", "assets/media/fitness/squat-demo.gif", start_sec=1.0, duration=2.6, fps_out=6, width=360, max_frames=18))
    paths["fitness_gif_pushup"] = rel(make_video_gif(fitness / "视频" / "out-宽距俯卧撑-正面.mp4", "assets/media/fitness/pushup-demo.gif", start_sec=1.0, duration=2.6, fps_out=6, width=360, max_frames=18))

    quant_pkg = SOURCE_ROOT / "量化系统正式版V1" / "portfolio_pack"
    quant_assets = quant_pkg / "素材包"
    paths["quant_latest"] = rel(copy_image(quant_assets / "最新预测.png", "assets/images/quant/latest-prediction.jpg", 1200, 760))
    paths["quant_sim"] = rel(copy_image(quant_assets / "真实模拟.png", "assets/images/quant/real-simulation.jpg", 1200, 760))
    eval_chart, top_chart = draw_quant_charts(
        quant_pkg / "assets" / "data" / "latest_model_evaluation.csv",
        quant_pkg / "assets" / "data" / "latest_model_top3.csv",
    )
    paths["quant_eval"] = rel(eval_chart)
    paths["quant_top3"] = rel(top_chart)

    rail = SOURCE_ROOT / "铁路部件拼接项目"
    rail_common = rail / "portfolio_pack" / "common_assets"
    paths["rail_final"] = rel(copy_image(rail_common / "ours_panorama_preview.jpg", "assets/images/railway/final-panorama.jpg", 1400, 780))
    paths["rail_frame1"] = rel(copy_image(rail_common / "frame_01.jpg", "assets/images/railway/frame-01.jpg", 900, 620))
    paths["rail_frame29"] = rel(copy_image(rail_common / "frame_29.jpg", "assets/images/railway/frame-29.jpg", 900, 620))
    paths["rail_frame58"] = rel(copy_image(rail_common / "frame_58.jpg", "assets/images/railway/frame-58.jpg", 900, 620))
    paths["rail_video_gif"] = rel(make_video_gif(rail / "铁路部件识别项目（视频文件转图像拼接）" / "展示资料" / "无人机视频.mp4", "assets/media/railway/uav-preview.gif", start_sec=2.0, duration=2.6, fps_out=6, width=460, max_frames=18))
    paths["rail_reveal"] = rel(make_reveal_gif(rail_common / "ours_panorama_preview.jpg", "assets/media/railway/panorama-build.gif", width=720))

    logistics = SOURCE_ROOT / "物流大数据垂直大模型" / "portfolio_pack" / "web_showcase_assets" / "assets"
    for name, key in [
        ("architecture_diagram.png", "log_arch"),
        ("dashboard_overview.png", "log_dashboard"),
        ("route_optimization_demo.png", "log_route"),
        ("rag_skill_trace.png", "log_rag"),
    ]:
        paths[key] = rel(copy_image(logistics / name, f"assets/images/logistics/{key}.jpg", 1200, 760))

    campus = SOURCE_ROOT / "校园网约车项目" / "portfolio_pack" / "素材包"
    shuy = SOURCE_ROOT / "舒腰健脊App开发" / "portfolio_pack" / "素材包"
    mountain = SOURCE_ROOT / "登山协会小程序" / "portfolio_pack" / "素材包"
    mini_srcs = [
        campus / "截图" / "01-首页_拼车调度台.png",
        shuy / "截图" / "患者端-首页.jpg",
        mountain / "截图" / "首页.jpg",
    ]
    paths["mini_collage"] = rel(make_miniapp_collage(mini_srcs, "assets/images/miniapps/miniapp-collage.jpg"))
    mini_groups = {
        "campus": [campus / "截图" / p for p in [
            "01-首页_拼车调度台.png", "03-发起拼单_费用预估.png", "04-订单详情_成团进度.png", "05-我的行程_待出行.png"
        ]],
        "shuy": [shuy / "截图" / p for p in [
            "患者端-首页.jpg", "患者端-训练.jpg", "患者端-评估.jpg", "医生端-工作台.jpg"
        ]],
        "mountain": [mountain / "截图" / p for p in ["首页.jpg", "活动.jpg", "服务.jpg", "公告.jpg"]],
    }
    for group, srcs in mini_groups.items():
        for idx, src in enumerate(srcs, 1):
            key = f"mini_{group}_{idx}"
            paths[key] = rel(copy_image(src, f"assets/images/miniapps/{key}.jpg", 600, 960))
    for src, key in [
        (campus / "视频" / "校园网约车.mp4", "mini_campus_video"),
        (shuy / "视频" / "舒腰健脊.mp4", "mini_shuy_video"),
        (mountain / "视频" / "登山协会.mp4", "mini_mountain_video"),
    ]:
        paths[key] = rel(copy_static(src, f"assets/media/miniapps/{key}.mp4"))

    return paths


def write_css_js() -> None:
    css = r"""
:root {
  --ink: #172027;
  --muted: #5f6b73;
  --line: #dde4e8;
  --panel: #ffffff;
  --soft: #f5f7f8;
  --teal: #1f7a78;
  --rust: #b25c2f;
  --gold: #b48a2c;
  --green: #4f7d4e;
  --shadow: 0 12px 28px rgba(21, 33, 40, .09);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: #fbfcfc;
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif;
  line-height: 1.58;
}
a { color: inherit; text-decoration: none; }
img, video { max-width: 100%; display: block; }
.site-header {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  min-height: 62px;
  padding: 0 32px;
  border-bottom: 1px solid rgba(20, 31, 38, .1);
  background: rgba(255,255,255,.94);
  backdrop-filter: blur(12px);
}
.brand { font-weight: 800; letter-spacing: 0; white-space: nowrap; }
nav { display: flex; gap: 16px; align-items: center; overflow-x: auto; font-size: 14px; color: var(--muted); }
nav a { white-space: nowrap; padding: 8px 0; }
nav a:hover { color: var(--teal); }
main { width: min(1180px, calc(100vw - 36px)); margin: 0 auto; }
.site-footer {
  width: min(1180px, calc(100vw - 36px));
  margin: 44px auto 0;
  padding: 24px 0 34px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  display: flex;
  justify-content: space-between;
  gap: 16px;
  font-size: 14px;
}
.home-hero, .project-hero {
  min-height: 520px;
  display: grid;
  grid-template-columns: .88fr 1.12fr;
  align-items: center;
  gap: 32px;
  padding: 36px 0 28px;
}
.home-hero h1, .project-hero h1 {
  font-size: clamp(34px, 5vw, 58px);
  line-height: 1.04;
  margin: 8px 0 18px;
  letter-spacing: 0;
}
.kicker {
  color: var(--rust);
  font-weight: 800;
  margin: 0;
}
.lead {
  color: #36444c;
  font-size: 18px;
  margin: 0 0 22px;
}
.tag-row { display: flex; flex-wrap: wrap; gap: 8px; }
.tag-row span {
  border: 1px solid var(--line);
  background: #fff;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 13px;
  color: #40515b;
}
.hero-visual {
  min-width: 0;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: var(--shadow);
  background: var(--soft);
}
.hero-visual img, .hero-visual video { width: 100%; height: 420px; object-fit: cover; }
.home-visual-grid {
  display: grid;
  grid-template-columns: 1.25fr .75fr;
  gap: 10px;
  background: #fff;
  padding: 10px;
}
.home-visual-grid img { width: 100%; height: 200px; object-fit: cover; border-radius: 6px; }
.home-visual-grid img:first-child { grid-row: span 2; height: 410px; }
.section { padding: 24px 0; }
.section-title { display: flex; align-items: end; justify-content: space-between; gap: 16px; border-bottom: 1px solid var(--line); margin-bottom: 18px; }
.section h2 { font-size: 26px; margin: 0 0 10px; letter-spacing: 0; }
.section-title p { margin: 0 0 12px; color: var(--muted); }
.project-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}
.project-card, .media-card, .metric, .text-card, .phone-card {
  background: var(--panel);
  border: 1px solid rgba(22,32,38,.08);
  border-radius: 8px;
  box-shadow: 0 8px 22px rgba(21,33,40,.05);
}
.project-card { overflow: hidden; display: flex; flex-direction: column; }
.project-card img, .project-card video { width: 100%; height: 190px; object-fit: cover; }
.project-card-body { padding: 14px; display: flex; flex-direction: column; gap: 9px; flex: 1; }
.project-card h3 { margin: 0; font-size: 18px; line-height: 1.25; }
.project-card p { margin: 0; color: var(--muted); font-size: 14px; }
.card-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: auto; }
.card-meta span { font-size: 12px; color: #4f5d65; background: #f1f4f5; padding: 4px 7px; border-radius: 999px; }
.button-link { color: var(--teal); font-weight: 800; margin-top: 4px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}
.metric { padding: 13px 14px; }
.metric span { display: block; color: var(--muted); font-size: 13px; }
.metric strong { display: block; font-size: 24px; margin-top: 3px; }
.media-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}
.media-grid.two { grid-template-columns: repeat(2, 1fr); }
.media-card { overflow: hidden; }
.media-card img, .media-card video { width: 100%; height: 230px; object-fit: cover; background: #edf1f2; }
.media-card.tall img { height: 420px; object-fit: contain; background: #eef1f2; }
.media-card div { padding: 12px 13px; }
.media-card h4 { margin: 0 0 5px; font-size: 16px; }
.media-card p { margin: 0; color: var(--muted); font-size: 14px; }
.text-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.text-card { padding: 16px; }
.text-card h3 { margin: 0 0 8px; font-size: 18px; }
.text-card p, .text-card li { color: var(--muted); }
.text-card p { margin: 0; }
.text-card ul { padding-left: 18px; margin: 0; }
.auto-carousel { position: relative; min-height: 420px; background: #eef2f3; overflow: hidden; border-radius: 8px; }
.auto-carousel .slide { position: absolute; inset: 0; opacity: 0; transition: opacity .5s ease; margin: 0; }
.auto-carousel .slide.is-active { opacity: 1; }
.auto-carousel img { width: 100%; height: 100%; object-fit: cover; }
.auto-carousel figcaption {
  position: absolute;
  left: 14px;
  bottom: 14px;
  background: rgba(255,255,255,.9);
  padding: 6px 9px;
  border-radius: 6px;
  font-size: 13px;
}
.carousel-dots { position: absolute; right: 14px; bottom: 14px; display: flex; gap: 6px; }
.carousel-dots button { width: 9px; height: 9px; border-radius: 50%; border: 0; background: rgba(255,255,255,.55); padding: 0; }
.carousel-dots button.is-active { background: var(--teal); }
.process {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}
.process div { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 13px; }
.process strong { display: block; margin-bottom: 4px; }
.process span { color: var(--muted); font-size: 14px; }
.phone-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.phone-card { overflow: hidden; background: #f4f6f7; }
.phone-card img { width: 100%; height: 420px; object-fit: contain; padding: 8px; }
.phone-card p { margin: 0; padding: 10px 12px; background: #fff; color: var(--muted); font-size: 14px; }
.flow { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; align-items: stretch; }
.flow div { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 12px; position: relative; }
.flow div:not(:last-child)::after { content: "→"; position: absolute; right: -13px; top: 50%; transform: translateY(-50%); color: var(--rust); font-weight: 800; }
.flow strong { display: block; margin-bottom: 4px; }
.flow span { font-size: 13px; color: var(--muted); }
.wide-image { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 8px; box-shadow: var(--shadow); }
.wide-image img { width: 100%; border-radius: 6px; }
figure { margin: 0; }
figcaption { color: var(--muted); font-size: 13px; margin-top: 7px; }
.split {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  align-items: start;
}
.note {
  background: #fff;
  border-left: 4px solid var(--teal);
  padding: 13px 15px;
  color: #35444c;
}
@media (max-width: 980px) {
  .site-header { align-items: flex-start; flex-direction: column; padding: 12px 18px; }
  main, .site-footer { width: min(100vw - 24px, 760px); }
  .home-hero, .project-hero, .split { grid-template-columns: 1fr; min-height: auto; }
  .project-grid, .media-grid, .media-grid.two, .text-grid { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .process, .flow, .phone-grid { grid-template-columns: 1fr 1fr; }
  .flow div::after { display: none; }
  .hero-visual img, .hero-visual video { height: 300px; }
}
@media (max-width: 560px) {
  .metrics, .process, .flow, .phone-grid { grid-template-columns: 1fr; }
  .site-footer { flex-direction: column; }
  .home-visual-grid { grid-template-columns: 1fr; }
  .home-visual-grid img:first-child, .home-visual-grid img { height: 220px; }
}
"""
    (REPO / "assets" / "styles.css").write_text(css, encoding="utf-8")
    js = r"""
document.querySelectorAll('[data-carousel]').forEach((carousel) => {
  const slides = Array.from(carousel.querySelectorAll('.slide'));
  const dots = Array.from(carousel.querySelectorAll('.carousel-dots button'));
  if (slides.length <= 1) return;
  let index = 0;
  const show = (next) => {
    slides[index].classList.remove('is-active');
    dots[index]?.classList.remove('is-active');
    index = (next + slides.length) % slides.length;
    slides[index].classList.add('is-active');
    dots[index]?.classList.add('is-active');
  };
  dots.forEach((dot, i) => dot.addEventListener('click', () => show(i)));
  setInterval(() => show(index + 1), 3600);
});
"""
    (REPO / "assets" / "site.js").write_text(js, encoding="utf-8")
    (REPO / ".nojekyll").write_text("", encoding="utf-8")


def write_index(paths: dict[str, str]) -> None:
    cards = [
        ("360车载全景影像拼接", "工程车辆四路鱼眼相机 BEV 拼接，上机演示与多尺度环视输出。", paths["avm_demo_bev"], "projects/avm-360.html", ["OpenCV", "BEV", "标定"]),
        ("ACAF2025金融智能创新大赛", "基金申赎 7 日预测，三等奖，覆盖数据工程、特征融合和 LSTM-Attention。", paths["acaf_award"], "projects/acaf-finance.html", ["PyTorch", "金融预测", "竞赛"]),
        ("AI健身动作识别与评分", "基于摄像头姿态估计、关键点规则、自动计数与运动反馈的视觉原型。", paths["fitness_gif_squat"], "projects/ai-fitness.html", ["MediaPipe", "OpenCV", "运动健康"]),
        ("事件与情绪分析 / ETF量化辅助", "多因子 ETF 轮动模拟盘，叠加新闻、公告和社区情绪的信息增强层。", paths["quant_sim"], "projects/quant-agent.html", ["ETF", "Agent", "机器学习"]),
        ("铁路无人机全景拼接", "从铁路巡检视频抽帧、配准到全景底图生成，服务工业视觉预处理。", paths["rail_final"], "projects/railway-stitching.html", ["SIFT", "RANSAC", "无人机"]),
        ("物流大数据垂直大模型", "RAG + 工具调用 + 数据分析 Skills，覆盖运价、路径、风险和自动周报。", paths["log_dashboard"], "projects/logistics-llm.html", ["RAG", "Streamlit", "物流"]),
        ("微信小程序展示合集", "校园网约车、舒腰健脊、登山协会三个小程序的产品与工程展示。", paths["mini_collage"], "projects/miniapps.html", ["小程序", "Vue", "产品原型"]),
    ]
    card_html = ""
    for title, desc, img, href, tags in cards:
        meta = "".join(f"<span>{esc(t)}</span>" for t in tags)
        card_html += f"""
        <a class="project-card" href="{esc(href)}">
          {media_tag(img, title)}
          <div class="project-card-body">
            <h3>{esc(title)}</h3>
            <p>{esc(desc)}</p>
            <div class="card-meta">{meta}</div>
            <span class="button-link">查看项目详情</span>
          </div>
        </a>
        """
    visual = f"""
    <div class="home-visual-grid">
      {media_tag(paths["avm_demo_bev"], "360环视上机展示")}
      {media_tag(paths["quant_sim"], "量化真实模拟")}
      {media_tag(paths["mini_collage"], "小程序合集")}
    </div>
    """
    body = f"""
    <section class="home-hero">
      <div class="hero-copy">
        <p class="kicker">Portfolio Overview</p>
        <h1>王俊松项目作品集</h1>
        <p class="lead">这里汇总我在计算机视觉、计算成像、金融智能、RAG 应用和微信小程序方向的代表项目。每个项目都保留关键截图、结果图和数据事实，便于快速了解项目目标、核心做法与可展示成果。</p>
        <div class="tag-row">
          <span>Computer Vision</span><span>Financial AI</span><span>RAG / Agent</span><span>Mini Program</span><span>Python / OpenCV / PyTorch</span>
        </div>
      </div>
      <div class="hero-visual">{visual}</div>
    </section>
    {section("项目总览", f'<div class="project-grid">{card_html}</div>')}
    """
    (REPO / "index.html").write_text(page_shell("项目作品集概览", "王俊松项目作品集总览", body), encoding="utf-8")


def write_avm(paths: dict[str, str]) -> None:
    visual = carousel([
        (f"../{paths['avm_demo_bev']}", "上机鸟瞰图"),
        (f"../{paths['avm_demo_front']}", "前向相机上机画面"),
        (f"../{paths['avm_demo_left']}", "左侧相机上机画面"),
        (f"../{paths['avm_demo_right']}", "右侧相机上机画面"),
        (f"../{paths['avm_demo_back']}", "后向相机上机画面"),
    ])
    body = hero(
        "360车载全景影像拼接",
        "工程车辆四路鱼眼相机 BEV 拼接",
        "基于前、后、左、右四路鱼眼相机图像完成去畸变、地面单应性变换、有效区域融合和上机展示，形成可复盘的装载机场景环视拼接结果。",
        visual,
        ["OpenCV", "Fisheye", "Homography", "BEV Fusion"],
    )
    body += section("相关参数", metric_cards([
        ("输入相机", "front / left / right / back"),
        ("标定点", "每路 8 个外角点"),
        ("back 重投影误差", "平均 1.352px"),
        ("输出画布", "1216x1436 / 1600x1800 / 2000x2300"),
    ]))
    body += section("上机演示图片", f"""
      <div class="media-grid">
        {image_card(f"../{paths['avm_demo_front']}", "前向视角")}
        {image_card(f"../{paths['avm_demo_left']}", "左侧视角")}
        {image_card(f"../{paths['avm_demo_right']}", "右侧视角")}
        {image_card(f"../{paths['avm_demo_back']}", "后向视角")}
        {image_card(f"../{paths['avm_demo_bev']}", "鸟瞰展示")}
      </div>
    """)
    body += section("BEV 拼接结果", f"""
      <div class="media-grid two">
        {image_card(f"../{paths['avm_baseline']}", "baseline 距离权重融合")}
        {image_card(f"../{paths['avm_owned']}", "ownership-prior 区域融合")}
        {image_card(f"../{paths['avm_safe']}", "extended safe 输出")}
        {image_card(f"../{paths['avm_wide']}", "extended wide 输出")}
      </div>
    """)
    body += section("算法流程", """
      <div class="process">
        <div><strong>1. 相机输入</strong><span>四路鱼眼图像与标定板点位进入统一配置。</span></div>
        <div><strong>2. 几何映射</strong><span>鱼眼去畸变后通过地面单应性投影到 BEV 坐标。</span></div>
        <div><strong>3. 区域融合</strong><span>使用有效区域 mask、距离权重和 ownership-prior 降低重影。</span></div>
        <div><strong>4. 展示输出</strong><span>生成多尺度 BEV 结果并结合上机画面完成展示。</span></div>
      </div>
    """)
    (PROJECTS_DIR / "avm-360.html").write_text(page_shell("360车载全景影像拼接", "装载机四路鱼眼相机 BEV 拼接项目", body, active="avm"), encoding="utf-8")


def write_acaf(paths: dict[str, str]) -> None:
    visual = carousel([
        (f"../{paths['acaf_roadshow']}", "路演现场"),
        (f"../{paths['acaf_award']}", "颁奖仪式"),
        (f"../{paths['acaf_trophy']}", "奖金奖杯"),
    ])
    body = hero(
        "AFAC2025金融智能创新大赛",
        "基金申赎长周期预测",
        "围绕 20 只基金未来 7 日申购与赎回预测，完成多源特征构建、LSTM-Attention 双通道建模、预测结果输出和路演交付，项目获得 AFAC2025 金融智能创新大赛三等奖。",
        visual,
        ["PyTorch", "LSTM-Attention", "多源特征", "队长"],
    )
    body += section("赛程数据", metric_cards([
        ("基金数量", "20 只"),
        ("训练集规模", "9460 行 / 18 列"),
        ("缺失值", "0"),
        ("预测结果", "140 条"),
    ]))
    body += section("数据与模型流程", f"""
      <div class="media-grid two">
        {image_card(f"../{paths['acaf_arch']}", "数据流与模型结构")}
        {image_card(f"../{paths['acaf_pred']}", "7 日申购/赎回预测汇总")}
        {image_card(f"../{paths['acaf_top10']}", "基金预测结果 Top10")}
        {image_card(f"../{paths['acaf_monthly']}", "训练数据月度变化")}
      </div>
    """)
    body += section("证书展示", f"""
      <div class="split">
        <div class="wide-image">{media_tag(f"../{paths['acaf_certificate']}", "获奖证书")}</div>
        <div class="text-card">
          <h3>项目职责</h3>
          <ul>
            <li>统筹数据工程、特征构建、模型调研和路演材料。</li>
            <li>将赛方申赎曝光、日历、百度指数、市场指数、投资者情绪和发薪特征对齐到基金/日期维度。</li>
            <li>参与申购/赎回双通道模型设计，用 7 天历史窗口预测未来 7 天资金流。</li>
          </ul>
        </div>
      </div>
    """)
    (PROJECTS_DIR / "acaf-finance.html").write_text(page_shell("AFAC2025金融智能创新大赛", "基金申赎 7 日预测项目", body, active="acaf"), encoding="utf-8")


def write_fitness(paths: dict[str, str]) -> None:
    body = hero(
        "AI健身动作识别与评分系统",
        "实时姿态估计与运动反馈",
        "基于摄像头画面进行人体姿态估计、关键点提取、动作规则建模、自动计数和评分反馈，覆盖深蹲、俯卧撑、引体向上、仰卧起坐等运动场景。",
        media_tag(f"../{paths['fitness_home']}", "AI健身首页图"),
        ["MediaPipe Pose", "OpenCV", "动作计数", "运动健康"],
    )
    body += section("项目展示", f"""
      <div class="media-grid two">
        {image_card(f"../{paths['fitness_gif_squat']}", "深蹲识别动图", "实时骨架叠加、角度判断和计数反馈。")}
        {image_card(f"../{paths['fitness_gif_pushup']}", "俯卧撑识别动图", "通过阶段状态机降低单帧阈值误触发。")}
      </div>
    """)
    body += section("动作截图", f"""
      <div class="media-grid">
        {image_card(f"../{paths['fitness_squat']}", "深蹲")}
        {image_card(f"../{paths['fitness_pushup']}", "俯卧撑")}
        {image_card(f"../{paths['fitness_pullup']}", "引体向上")}
        {image_card(f"../{paths['fitness_situp']}", "仰卧起坐")}
      </div>
    """)
    body += section("识别链路", """
      <div class="process">
        <div><strong>1. 视频输入</strong><span>摄像头或本地视频作为实时/离线处理入口。</span></div>
        <div><strong>2. 姿态估计</strong><span>提取肩、肘、腕、髋、膝、踝等关键点。</span></div>
        <div><strong>3. 规则判别</strong><span>用关节角度、动作阶段和状态机完成计数。</span></div>
        <div><strong>4. 反馈展示</strong><span>叠加骨架、分数、次数、热量和训练反馈。</span></div>
      </div>
    """)
    (PROJECTS_DIR / "ai-fitness.html").write_text(page_shell("AI健身动作识别与评分系统", "姿态估计、动作计数与运动反馈项目", body, active="fitness"), encoding="utf-8")


def write_quant(paths: dict[str, str]) -> None:
    body = hero(
        "事件与情绪分析智能体 / ETF量化辅助系统",
        "多因子量化主引擎 + 信息增强层",
        "系统以 ETF 横截面轮动模型为主引擎，结合事件新闻情绪感知 Agent 生成事件对象、板块情绪和 ETF 信息增强画像，用于模型输出后的确认、降权、风险过滤和解释。",
        media_tag(f"../{paths['quant_latest']}", "最新预测截图"),
        ["ETF轮动", "Logistic / XGBoost / LightGBM", "事件情绪 Agent", "Tkinter"],
    )
    body += section("系统数据", metric_cards([
        ("ETF 因子记录", "62835 行"),
        ("交易日", "1781 个"),
        ("入模特征", "46 个"),
        ("V2 事件对象", "2145 条"),
    ]))
    body += section("量化预测与模拟", f"""
      <div class="media-grid two">
        {image_card(f"../{paths['quant_latest']}", "最新预测界面", "查看模型预测、策略目标和候选 ETF 排序。")}
        {image_card(f"../{paths['quant_sim']}", "真实模拟", "从一个观察日开始，按所选策略每10个交易日滚动调仓，模拟资金变化。")}
        {image_card(f"../{paths['quant_top3']}", "最新 Top3 概率")}
        {image_card(f"../{paths['quant_eval']}", "模型评估示例")}
      </div>
    """)
    body += section("事件与情绪增强层", """
      <div class="text-grid">
        <div class="text-card"><h3>数据入口</h3><p>接入财经新闻、专业披露与会议纪要、基金社区评论等真实公开源，形成每日原始汇总和统一证据对象。</p></div>
        <div class="text-card"><h3>结构化处理</h3><p>通过质量评分、跨源聚类、事件本体映射、LLM 结构化增强和规则回退，生成事件对象与板块情绪。</p></div>
        <div class="text-card"><h3>量化承接</h3><p>事件/情绪层不替代量化排序，而用于解释模型结果、识别负面风险、提示板块热度和辅助复盘。</p></div>
      </div>
    """)
    (PROJECTS_DIR / "quant-agent.html").write_text(page_shell("事件与情绪分析智能体 / ETF量化辅助系统", "ETF 轮动、事件情绪和模拟决策系统", body, active="quant"), encoding="utf-8")


def write_railway(paths: dict[str, str]) -> None:
    body = hero(
        "铁路无人机视频全景拼接",
        "工业巡检视频到全景底图",
        "面向铁路巡检视频视野碎片化的问题，完成视频抽帧、特征匹配、几何配准、全局位姿链和全景输出，为后续部件识别和巡检勘察提供大视野底图。",
        media_tag(f"../{paths['rail_final']}", "铁路全景拼接结果"),
        ["OpenCV", "SIFT / RootSIFT", "RANSAC", "无人机影像"],
    )
    body += section("项目状态", metric_cards([
        ("输入帧", "58 张 1920x1080"),
        ("匹配边日志", "227 条"),
        ("位姿链", "58 帧"),
        ("全景输出", "13567x11902"),
    ]))
    body += section("视频与拼接过程", f"""
      <div class="media-grid two">
        {image_card(f"../{paths['rail_video_gif']}", "原始无人机视频片段", "连续航拍画面作为拼接输入。")}
        {image_card(f"../{paths['rail_reveal']}", "全景逐步成图效果", "用逐步展开的方式展示全景结果的形成。")}
      </div>
    """)
    body += section("关键帧与结果", f"""
      <div class="media-grid">
        {image_card(f"../{paths['rail_frame1']}", "输入帧 01")}
        {image_card(f"../{paths['rail_frame29']}", "输入帧 29")}
        {image_card(f"../{paths['rail_frame58']}", "输入帧 58")}
      </div>
      <div class="wide-image" style="margin-top:14px">{media_tag(f"../{paths['rail_final']}", "全景输出")}</div>
    """)
    body += section("技术路线", """
      <div class="process">
        <div><strong>1. 视频抽帧</strong><span>从无人机视频中筛选连续帧并整理为拼接序列。</span></div>
        <div><strong>2. 特征匹配</strong><span>SIFT/RootSIFT 提取局部特征，KNN 生成候选匹配。</span></div>
        <div><strong>3. 几何配准</strong><span>RANSAC 估计仿射关系，ECC 处理弱连接场景。</span></div>
        <div><strong>4. 全景输出</strong><span>基于全局位姿链生成铁路场景大视野底图。</span></div>
      </div>
    """)
    body += section("成果说明", """
      <div class="note">当前页面展示的是图像拼接与巡检底图生成能力，核心价值在于把碎片化视频帧转化为可观察、可复盘的大视野场景。后续可以继续接入部件标注、目标检测和地理定位信息，形成更完整的铁路视觉巡检链路。</div>
    """)
    (PROJECTS_DIR / "railway-stitching.html").write_text(page_shell("铁路无人机视频全景拼接", "铁路巡检视频全景拼接项目", body, active="rail"), encoding="utf-8")


def write_logistics(paths: dict[str, str]) -> None:
    body = hero(
        "物流大数据垂直大模型应用原型",
        "RAG + 工具调用 + 数据分析 Skills",
        "面向物流公共服务平台中的智能问答、运价分析、路径优化、风险预警和自动周报场景，构建一个可离线演示的垂直大模型应用原型。",
        media_tag(f"../{paths['log_dashboard']}", "物流运行态势面板"),
        ["RAG", "Streamlit", "Dijkstra", "自动周报"],
    )
    body += section("数据资产", metric_cards([
        ("模拟运单", "1766 条"),
        ("仓储记录", "222 条"),
        ("节点风险", "296 条"),
        ("路线边", "32 条"),
    ]))
    body += section("系统展示", f"""
      <div class="media-grid two">
        {image_card(f"../{paths['log_arch']}", "项目总体架构")}
        {image_card(f"../{paths['log_dashboard']}", "运行态势面板")}
        {image_card(f"../{paths['log_route']}", "路径优化演示")}
        {image_card(f"../{paths['log_rag']}", "RAG 与技能调用链")}
      </div>
    """)
    body += section("应用能力", """
      <div class="text-grid">
        <div class="text-card"><h3>结构化数据分析</h3><p>运单、仓储、节点和路网数据进入指标层，用于运价、风险、需求和路径类任务。</p></div>
        <div class="text-card"><h3>知识库检索</h3><p>政策、冷链 SOP、铁路通道、风险处置和数据治理文档统一进入检索链路。</p></div>
        <div class="text-card"><h3>领域工具调用</h3><p>金额、路径、预测和风险判断交给确定性 Skills 执行，再由应用层组织成可读报告。</p></div>
      </div>
    """)
    (PROJECTS_DIR / "logistics-llm.html").write_text(page_shell("物流大数据垂直大模型应用原型", "物流 RAG 与工具调用系统", body, active="logistics"), encoding="utf-8")


def write_miniapps(paths: dict[str, str]) -> None:
    def phone_cards(keys: list[str], labels: list[str]) -> str:
        return "".join(f'<div class="phone-card">{media_tag(f"../{paths[key]}", labels[i])}<p>{esc(labels[i])}</p></div>' for i, key in enumerate(keys))

    body = hero(
        "微信小程序展示合集",
        "校园网约车 / 舒腰健脊 / 登山协会",
        "三个小程序覆盖校园出行、康复健康管理和社团活动服务，重点展示从需求拆解、页面流程、状态联动到可运行 Demo 的产品工程能力。",
        media_tag(f"../{paths['mini_collage']}", "小程序合集"),
        ["微信小程序", "uni-app / Vue 3", "产品原型", "状态联动"],
    )
    body += section("统一产品链路", """
      <div class="flow">
        <div><strong>需求拆解</strong><span>把真实场景拆成角色、任务和页面入口。</span></div>
        <div><strong>数据建模</strong><span>用本地 Mock / storage 承接用户、订单、训练、随访和活动数据。</span></div>
        <div><strong>页面实现</strong><span>按主流程组织首页、详情、表单、记录和个人中心。</span></div>
        <div><strong>交互闭环</strong><span>让创建、加入、打卡、评价、随访和报名动作产生状态联动。</span></div>
        <div><strong>展示交付</strong><span>用截图与录屏展示端到端体验。</span></div>
      </div>
    """)
    body += section("校园网约车项目", f"""
      <div class="split">
        <div class="media-card">{media_tag(f"../{paths['mini_campus_video']}", "校园网约车演示视频", poster=f"../{paths['mini_campus_1']}")}<div><h4>演示视频</h4><p>覆盖找单、发起拼单、订单详情、行程管理和历史评价。</p></div></div>
        <div class="text-card"><h3>核心流程</h3><ul><li>首页找单、路线筛选、发起拼单和订单详情。</li><li>本地 Mock 数据驱动成团进度、座位、费用和司机状态。</li><li>验证 H5 与微信小程序构建，适合展示跨端原型能力。</li></ul></div>
      </div>
      <div class="phone-grid" style="margin-top:14px">
        {phone_cards(['mini_campus_1','mini_campus_2','mini_campus_3','mini_campus_4'], ['拼车调度台','发起拼单','订单详情','我的行程'])}
      </div>
    """)
    body += section("舒腰健脊 App / 小程序", f"""
      <div class="split">
        <div class="media-card">{media_tag(f"../{paths['mini_shuy_video']}", "舒腰健脊演示视频", poster=f"../{paths['mini_shuy_1']}")}<div><h4>演示视频</h4><p>患者端与诊疗师端双角色健康管理原型。</p></div></div>
        <div class="text-card"><h3>核心流程</h3><ul><li>患者端：首页、评估、训练、随访和个人中心。</li><li>诊疗师端：工作台、方案、随访和患者管理。</li><li>围绕康复训练、疼痛记录和问诊报告构建状态联动。</li></ul></div>
      </div>
      <div class="phone-grid" style="margin-top:14px">
        {phone_cards(['mini_shuy_1','mini_shuy_2','mini_shuy_3','mini_shuy_4'], ['患者端首页','训练课程','问诊评估','医生工作台'])}
      </div>
    """)
    body += section("登山协会小程序", f"""
      <div class="split">
        <div class="media-card">{media_tag(f"../{paths['mini_mountain_video']}", "登山协会演示视频", poster=f"../{paths['mini_mountain_1']}")}<div><h4>演示视频</h4><p>围绕社团活动、公告、服务和个人中心组织信息展示。</p></div></div>
        <div class="text-card"><h3>核心流程</h3><ul><li>首页聚合活动入口和协会信息。</li><li>活动、公告、服务和我的页面形成清晰导航。</li><li>适合展示轻量组织类小程序的信息架构和视觉落地。</li></ul></div>
      </div>
      <div class="phone-grid" style="margin-top:14px">
        {phone_cards(['mini_mountain_1','mini_mountain_2','mini_mountain_3','mini_mountain_4'], ['首页','活动','服务','公告'])}
      </div>
    """)
    (PROJECTS_DIR / "miniapps.html").write_text(page_shell("微信小程序展示合集", "校园网约车、舒腰健脊与登山协会小程序", body, active="miniapps"), encoding="utf-8")


def write_readme() -> None:
    readme = """# 王俊松项目作品集

这是一个 GitHub Pages 静态作品集站点，入口为 `index.html`。

公开访问地址：

```text
https://wjs2000.github.io/show/
```

站点内容包括：

- 360车载全景影像拼接
- AFAC2025金融智能创新大赛
- AI健身动作识别与评分系统
- 事件与情绪分析智能体 / ETF量化辅助系统
- 铁路无人机视频全景拼接
- 物流大数据垂直大模型应用原型
- 微信小程序展示合集
"""
    (REPO / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_clean()
    paths = write_assets()
    write_css_js()
    write_index(paths)
    write_avm(paths)
    write_acaf(paths)
    write_fitness(paths)
    write_quant(paths)
    write_railway(paths)
    write_logistics(paths)
    write_miniapps(paths)
    write_readme()
    print("Built portfolio site")


if __name__ == "__main__":
    main()
