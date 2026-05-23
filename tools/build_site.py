# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import html
import ast
import json
import math
import os
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
ASSET_VERSION = "20260523_quant_layout"


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


def copy_image(src: Path, dest_rel: str, max_w: int = 1600, max_h: int = 1000, quality: int = 88) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dest = slug_path(dest_rel)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        if dest.suffix.lower() in [".jpg", ".jpeg", ".webp"] and im.mode == "RGBA":
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
    if not src.exists():
        raise FileNotFoundError(src)
    dest = slug_path(dest_rel)
    shutil.copy2(src, dest)
    return dest


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc") if bold else Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for item in candidates:
        if item.exists():
            return ImageFont.truetype(str(item), size)
    return ImageFont.load_default()


def make_video_gif(
    src: Path,
    dest_rel: str,
    start_sec: float = 0,
    duration: float = 3.0,
    fps_out: int = 6,
    width: int = 360,
    max_frames: int = 18,
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
            new_h = max(1, int(h * width / w))
            frame = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(frame).convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
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


def make_reveal_gif(src: Path, dest_rel: str, width: int = 760, frames_count: int = 30) -> Path:
    dest = slug_path(dest_rel)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        ratio = width / im.width
        height = max(1, int(im.height * ratio))
        im = im.resize((width, height), Image.Resampling.LANCZOS)
    frames: list[Image.Image] = []
    bg = Image.new("RGB", im.size, (10, 13, 15))
    grid = ImageDraw.Draw(bg)
    for x in range(0, width, 48):
        grid.line([(x, 0), (x, height)], fill=(28, 34, 38))
    for y in range(0, height, 48):
        grid.line([(0, y), (width, y)], fill=(28, 34, 38))
    for i in range(frames_count):
        t = (i + 1) / frames_count
        reveal_w = int(width * t)
        frame = bg.copy()
        frame.paste(im.crop((0, 0, reveal_w, height)), (0, 0))
        draw = ImageDraw.Draw(frame)
        draw.line([(reveal_w, 0), (reveal_w, height)], fill=(57, 177, 188), width=3)
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    frames.extend([frames[-1]] * 8)
    frames[0].save(dest, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    return dest


def make_media_strip(srcs: list[Path], dest_rel: str, labels: list[str], max_h: int = 560) -> Path:
    dest = slug_path(dest_rel)
    panels: list[Image.Image] = []
    for src, label in zip(srcs, labels):
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((520, max_h - 58), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (540, max_h), (246, 249, 251))
            x = (canvas.width - im.width) // 2
            y = 18
            canvas.paste(im, (x, y))
            draw = ImageDraw.Draw(canvas)
            draw.text((22, max_h - 38), label, fill=(22, 32, 40), font=font(20, True))
            panels.append(canvas)
    gap = 22
    out = Image.new("RGB", (len(panels) * 540 + (len(panels) - 1) * gap, max_h), (236, 242, 247))
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += 540 + gap
    out.thumbnail((1800, 720), Image.Resampling.LANCZOS)
    out.save(dest, "JPEG", quality=88, optimize=True, progressive=True)
    return dest


def make_miniapp_collage(srcs: list[Path], dest_rel: str) -> Path:
    dest = slug_path(dest_rel)
    thumbs = []
    for src in srcs:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((300, 610), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (330, 650), (244, 248, 249))
            x = (canvas.width - im.width) // 2
            y = (canvas.height - im.height) // 2
            canvas.paste(im, (x, y))
            thumbs.append(canvas)
    w = sum(im.width for im in thumbs) + 26 * (len(thumbs) - 1)
    h = max(im.height for im in thumbs)
    out = Image.new("RGB", (w, h), (223, 237, 239))
    x = 0
    for i, im in enumerate(thumbs):
        y = 0 if i % 2 == 0 else 24
        out.paste(im, (x, y))
        x += im.width + 26
    out.thumbnail((1400, 780), Image.Resampling.LANCZOS)
    out.save(dest, "JPEG", quality=88, optimize=True, progressive=True)
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
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=170)
    x = np.arange(len(models))
    ax.bar(x - 0.18, auc, 0.34, label="Test AUC", color="#2f7a84")
    ax.bar(x + 0.18, p3, 0.34, label="P@3", color="#c2772e")
    ax.set_xticks(x, models)
    ymin = max(0.48, min(auc + p3) - 0.02)
    ymax = min(0.62, max(auc + p3) + 0.03)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.24)
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
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=170)
    ypos = np.arange(len(names))
    ax.barh(ypos, probs, color=["#246d7d", "#55936b", "#be7a2d"])
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlim(max(0.50, min(probs) - 0.025), min(0.62, max(probs) + 0.035))
    ax.grid(axis="x", alpha=0.24)
    ax.set_title("Latest Logistic Top 3 Probability")
    for y, value in zip(ypos, probs):
        ax.text(value + 0.002, y, f"{value:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(chart2, bbox_inches="tight")
    plt.close(fig)
    return chart1, chart2


def draw_blue_tech_background(dest_rel: str) -> Path:
    dest = slug_path(dest_rel)
    w, h = 1800, 980
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            t = x / w
            u = y / h
            arr[y, x] = [
                int(16 + 18 * t + 10 * u),
                int(62 + 52 * t + 24 * u),
                int(110 + 86 * (1 - u) + 28 * t),
            ]
    im = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    for x in range(80, w, 140):
        draw.line([(x, 0), (x - 360, h)], fill=(180, 230, 255, 24), width=1)
    for y in range(80, h, 120):
        draw.line([(0, y), (w, y + 120)], fill=(180, 230, 255, 15), width=1)
    for cx, cy, r in [(1380, 270, 210), (1170, 640, 160), (1500, 700, 120)]:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(175, 229, 255, 46), width=3)
    for i in range(9):
        x = 220 + i * 145
        y = 680 - int(50 * math.sin(i * 0.7))
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(197, 245, 255, 120))
        if i:
            px = 220 + (i - 1) * 145
            py = 680 - int(50 * math.sin((i - 1) * 0.7))
            draw.line([(px, py), (x, y)], fill=(197, 245, 255, 62), width=3)
    im.save(dest, "JPEG", quality=88, optimize=True, progressive=True)
    return dest


def draw_quant_background(dest_rel: str) -> Path:
    dest = slug_path(dest_rel)
    w, h = 1800, 980
    im = Image.new("RGB", (w, h), (9, 18, 23))
    draw = ImageDraw.Draw(im, "RGBA")
    for x in range(0, w, 90):
        draw.line([(x, 0), (x, h)], fill=(230, 246, 245, 18), width=1)
    for y in range(70, h, 85):
        draw.line([(0, y), (w, y)], fill=(230, 246, 245, 14), width=1)
    rng = np.random.default_rng(42)
    price = 510
    x = 760
    for i in range(38):
        open_p = price + int(rng.normal(0, 28))
        close_p = open_p + int(rng.normal(0, 42))
        high = max(open_p, close_p) + int(rng.uniform(16, 58))
        low = min(open_p, close_p) - int(rng.uniform(16, 58))
        color = (230, 82, 82, 190) if close_p >= open_p else (43, 192, 128, 190)
        y_high = h - high
        y_low = h - low
        y_open = h - open_p
        y_close = h - close_p
        draw.line([(x, y_high), (x, y_low)], fill=color, width=3)
        draw.rounded_rectangle((x - 14, min(y_open, y_close), x + 14, max(y_open, y_close)), radius=3, fill=color)
        price = close_p
        x += 28
    points = []
    for i in range(34):
        points.append((760 + i * 32, 500 - 85 * math.sin(i / 4.0) + rng.normal(0, 18)))
    draw.line(points, fill=(255, 196, 82, 150), width=4)
    draw.rectangle((0, 0, w, h), fill=(0, 0, 0, 18))
    im.save(dest, "JPEG", quality=90, optimize=True, progressive=True)
    return dest


def draw_quant_agent_background(dest_rel: str) -> Path:
    dest = slug_path(dest_rel)
    w, h = 1800, 980
    im = Image.new("RGB", (w, h), (7, 14, 19))
    draw = ImageDraw.Draw(im, "RGBA")
    for x in range(0, w, 88):
        draw.line([(x, 0), (x, h)], fill=(206, 236, 235, 16), width=1)
    for y in range(70, h, 86):
        draw.line([(0, y), (w, y)], fill=(206, 236, 235, 13), width=1)

    rng = np.random.default_rng(7)
    price = 510
    x = 820
    for _ in range(34):
        open_p = price + int(rng.normal(0, 22))
        close_p = open_p + int(rng.normal(0, 35))
        high = max(open_p, close_p) + int(rng.uniform(12, 50))
        low = min(open_p, close_p) - int(rng.uniform(12, 50))
        color = (230, 86, 82, 165) if close_p >= open_p else (35, 188, 124, 165)
        y_high = h - high
        y_low = h - low
        y_open = h - open_p
        y_close = h - close_p
        draw.line([(x, y_high), (x, y_low)], fill=color, width=3)
        draw.rounded_rectangle((x - 13, min(y_open, y_close), x + 13, max(y_open, y_close)), radius=3, fill=color)
        price = close_p
        x += 29

    nodes = [
        (1040, 280, "新闻"),
        (1260, 215, "政策"),
        (1460, 340, "披露"),
        (1360, 560, "社区"),
        (1090, 650, "ETF"),
        (840, 470, "模型"),
        (1550, 690, "风险"),
    ]
    for i, (x1, y1, _) in enumerate(nodes):
        for x2, y2, _ in nodes[i + 1 :]:
            d = math.dist((x1, y1), (x2, y2))
            if d < 440:
                draw.line([(x1, y1), (x2, y2)], fill=(107, 214, 203, 42), width=2)
    label_font = font(22, True)
    for x, y, label in nodes:
        r = 28 if label in ["ETF", "模型"] else 22
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(46, 174, 150, 150), outline=(220, 255, 247, 130), width=2)
        draw.text((x - 22, y + r + 8), label, fill=(225, 246, 245, 150), font=label_font)

    points = []
    for i in range(38):
        points.append((760 + i * 30, 500 - 75 * math.sin(i / 4.2) + rng.normal(0, 15)))
    draw.line(points, fill=(255, 194, 88, 125), width=4)
    draw.rectangle((0, 0, w, h), fill=(0, 0, 0, 20))
    im.save(dest, "JPEG", quality=90, optimize=True, progressive=True)
    return dest


def draw_logistics_background(dest_rel: str) -> Path:
    dest = slug_path(dest_rel)
    w, h = 1800, 980
    im = Image.new("RGB", (w, h), (16, 38, 45))
    draw = ImageDraw.Draw(im, "RGBA")
    for y in range(h):
        alpha = int(120 * y / h)
        draw.line([(0, y), (w, y)], fill=(24, 82, 87, alpha), width=1)
    nodes = [(980, 250), (1250, 210), (1460, 350), (1350, 560), (1060, 660), (840, 470), (1560, 680)]
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if math.dist(a, b) < 470:
                draw.line([a, b], fill=(138, 224, 207, 42), width=2)
    for idx, (x, y) in enumerate(nodes):
        r = 18 if idx != 3 else 28
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(105, 220, 190, 155), outline=(230, 255, 248, 135), width=2)
    for x, y, ww, hh in [(1080, 785, 150, 80), (1260, 760, 210, 110), (1510, 785, 130, 70)]:
        draw.rounded_rectangle((x, y, x + ww, y + hh), radius=8, fill=(232, 197, 98, 115), outline=(255, 235, 150, 120), width=2)
        draw.line([(x + 18, y + 18), (x + ww - 18, y + hh - 18)], fill=(255, 235, 150, 60), width=2)
    im.save(dest, "JPEG", quality=90, optimize=True, progressive=True)
    return dest


def draw_miniapp_background(dest_rel: str) -> Path:
    dest = slug_path(dest_rel)
    w, h = 1800, 980
    im = Image.new("RGB", (w, h), (230, 244, 241))
    draw = ImageDraw.Draw(im, "RGBA")
    draw.rounded_rectangle((860, 120, 1650, 790), radius=24, fill=(18, 31, 38, 220))
    draw.rounded_rectangle((900, 170, 1260, 745), radius=16, fill=(247, 250, 250, 245))
    draw.rounded_rectangle((1290, 170, 1610, 745), radius=16, fill=(33, 47, 56, 245))
    for i, color in enumerate([(238, 86, 86), (244, 190, 77), (73, 190, 120)]):
        x = 895 + i * 20
        draw.ellipse((x, 138, x + 10, 148), fill=color)
    for i in range(10):
        y = 210 + i * 46
        draw.rounded_rectangle((925, y, 1228, y + 20), radius=4, fill=(37, 159, 112, 50))
        draw.rounded_rectangle((1320, y, 1570 - i * 9, y + 18), radius=4, fill=(132, 226, 183, 72))
    for x, y, scale in [(1040, 480, 1.0), (760, 560, 0.82), (1450, 520, 0.88)]:
        ww, hh = int(185 * scale), int(360 * scale)
        draw.rounded_rectangle((x, y, x + ww, y + hh), radius=int(32 * scale), fill=(255, 255, 255, 245), outline=(20, 42, 48, 80), width=3)
        draw.rounded_rectangle((x + 18, y + 48, x + ww - 18, y + hh - 28), radius=10, fill=(236, 248, 246, 255))
        draw.rounded_rectangle((x + 42, y + 82, x + ww - 42, y + 124), radius=10, fill=(37, 159, 112, 180))
        draw.rounded_rectangle((x + 42, y + 145, x + ww - 42, y + 174), radius=8, fill=(42, 67, 79, 40))
        draw.rounded_rectangle((x + 42, y + 190, x + ww - 42, y + 218), radius=8, fill=(42, 67, 79, 30))
    im.save(dest, "JPEG", quality=90, optimize=True, progressive=True)
    return dest


def media_tag(src: str, alt: str = "", cls: str = "", caption: str | None = None, poster: str | None = None) -> str:
    ext = Path(src).suffix.lower()
    cls_attr = f' class="{esc(cls)}"' if cls else ""
    if ext == ".mp4":
        poster_attr = f' poster="{esc(poster)}"' if poster else ""
        body = f'<video{cls_attr} controls muted playsinline preload="metadata"{poster_attr}><source src="{esc(src)}" type="video/mp4"></video>'
    else:
        body = f'<img{cls_attr} src="{esc(src)}" alt="{esc(alt)}" loading="lazy">'
    if caption:
        return f'<figure>{body}<figcaption>{esc(caption)}</figcaption></figure>'
    return body


def read_csv_dicts(src: Path) -> list[dict[str, str]]:
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str | float | int, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def reason_list(value: str, limit: int = 2) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed[:limit]]
    except (SyntaxError, ValueError):
        pass
    return [value] if value else []


def tags_html(tags: list[str]) -> str:
    return "".join(f"<span>{esc(tag)}</span>" for tag in tags)


def metric_cards(metrics: Iterable[tuple[str, str]]) -> str:
    return '<div class="metrics">' + "".join(
        f'<div class="metric"><strong>{esc(v)}</strong><span>{esc(k)}</span></div>' for k, v in metrics
    ) + "</div>"


def picture_panel(src: str, title: str, desc: str = "", cls: str = "") -> str:
    desc_html = f"<p>{esc(desc)}</p>" if desc else ""
    return f"""
    <article class="picture-panel {esc(cls)}">
      <div class="picture">{media_tag(src, title)}</div>
      <div class="picture-copy">
        <h3>{esc(title)}</h3>
        {desc_html}
      </div>
    </article>
    """


def text_panel(title: str, desc: str) -> str:
    return f'<article class="text-panel"><h3>{esc(title)}</h3><p>{esc(desc)}</p></article>'


def immersive_hero(
    title: str,
    kicker: str,
    desc: str,
    image_src: str,
    tags: list[str],
    *,
    contain: bool = False,
    tone: str = "",
) -> str:
    contain_cls = " hero-contain" if contain else ""
    tone_cls = f" {tone}" if tone else ""
    return f"""
    <section class="immersive-hero{contain_cls}{tone_cls}" style="--hero-image: url('{esc(image_src)}')">
      <div class="hero-bg" aria-hidden="true"></div>
      <div class="hero-shade" aria-hidden="true"></div>
      <div class="hero-inner">
        <p class="kicker">{esc(kicker)}</p>
        <h1>{esc(title)}</h1>
        <p class="lead">{esc(desc)}</p>
        <div class="tag-row">{tags_html(tags)}</div>
      </div>
    </section>
    """


def home_hero(slides: list[str]) -> str:
    slide_html = "".join(
        f'<figure class="hero-slide {"is-active" if i == 0 else ""}"><img src="{esc(src)}" alt="" loading="{"eager" if i == 0 else "lazy"}"></figure>'
        for i, src in enumerate(slides)
    )
    dots = "".join(f'<button class="{"is-active" if i == 0 else ""}" aria-label="切换首屏图 {i + 1}"></button>' for i in range(len(slides)))
    return f"""
    <section class="immersive-hero home-immersive" data-hero-carousel>
      <div class="hero-slides">{slide_html}</div>
      <div class="hero-shade" aria-hidden="true"></div>
      <div class="hero-inner">
        <p class="kicker">Portfolio Overview</p>
        <h1>项目作品集</h1>
        <p class="lead">本项目集为计算机视觉、计算成像、金融智能、RAG 应用和微信小程序方向的代表项目。</p>
        <div class="tag-row">
          <span>Computer Vision</span><span>Computational Imaging</span><span>Financial AI</span><span>RAG / Agent</span><span>Mini Program</span>
        </div>
      </div>
      <div class="hero-dots">{dots}</div>
    </section>
    """


def section(title: str, content: str, cls: str = "", subtitle: str = "") -> str:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    return f"""
    <section class="section {esc(cls)}">
      <div class="section-heading">
        <h2>{esc(title)}</h2>
        {sub}
      </div>
      {content}
    </section>
    """


def page_shell(title: str, subtitle: str, body: str, active: str = "") -> str:
    nav_items = [
        ("../index.html" if active else "index.html", "首页"),
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
  <title>{esc(title)} | 项目作品集</title>
  <meta name="description" content="{esc(subtitle)}">
  <link rel="stylesheet" href="{prefix}assets/styles.css?v={ASSET_VERSION}">
</head>
<body>
  <header class="site-header">
    <a class="brand" href="{prefix}index.html">项目作品集</a>
    <nav>{nav}</nav>
  </header>
  <main>{body}</main>
  <footer class="site-footer">
    <span>Computer Vision · Computational Imaging · Financial AI · RAG · Mini Programs</span>
  </footer>
  <script src="{prefix}assets/site.js?v={ASSET_VERSION}"></script>
</body>
</html>
"""


def write_assets() -> dict[str, str]:
    paths: dict[str, str] = {}

    avm_base = SOURCE_ROOT / "360车载全景影像拼接" / "portfolio_pack"
    avm_demo = avm_base / "素材展示" / "上机演示图片"
    avm_show = avm_base / "showcase_materials" / "resume_zip_showcase" / "images"
    for name, key in [
        ("01_front_raw.png", "avm_raw_front"),
        ("02_left_raw.png", "avm_raw_left"),
        ("03_right_raw.png", "avm_raw_right"),
        ("04_back_raw.png", "avm_raw_back"),
        ("05_baseline_bev.png", "avm_baseline"),
        ("06_owned_regions_best.png", "avm_owned"),
        ("07_extended_safe.png", "avm_safe"),
        ("08_extended_wide.png", "avm_wide"),
    ]:
        paths[key] = rel(copy_image(avm_show / name, f"assets/images/avm/{key}.jpg", 1500, 980))
    for name, key in [
        ("上机前.jpg", "avm_demo_front"),
        ("上机右.jpg", "avm_demo_right"),
        ("上机后.jpg", "avm_demo_back"),
        ("上机左.jpg", "avm_demo_left"),
        ("上机鸟瞰图.jpg", "avm_demo_bev"),
    ]:
        paths[key] = rel(copy_image(avm_demo / name, f"assets/images/avm/{key}.jpg", 1400, 900))
    avm_out = avm_base / "ZhuanFa" / "ZhuanFa" / "suanfa" / "04_outputs"
    for name, key in [
        ("formal_compare_before_after.png", "avm_latency_compare"),
        ("debug_process_time_breakdown.png", "avm_latency_breakdown"),
        ("formal_roi_optimized_output.png", "avm_roi_output"),
    ]:
        if (avm_out / name).exists():
            paths[key] = rel(copy_image(avm_out / name, f"assets/images/avm/{key}.jpg", 1300, 900))

    acaf_img = SOURCE_ROOT / "ACAF2025金融智能创新大赛" / "portfolio_pack" / "assets" / "github_pages_materials" / "images"
    for name, key in [
        ("路演现场.jpg", "acaf_roadshow"),
        ("颁奖仪式.jpg", "acaf_award"),
        ("奖金奖杯.jpg", "acaf_trophy"),
        ("证书.jpg", "acaf_certificate"),
    ]:
        paths[key] = rel(copy_image(acaf_img / name, f"assets/images/acaf/{key}.jpg", 1500, 980))
    acaf_charts = SOURCE_ROOT / "ACAF2025金融智能创新大赛" / "portfolio_pack" / "assets" / "charts"
    for name, key in [
        ("prediction_7day_aggregate.png", "acaf_pred"),
        ("fund_prediction_top10.png", "acaf_top10"),
        ("training_monthly_actuals.png", "acaf_monthly"),
        ("feature_correlation_snapshot.png", "acaf_corr"),
    ]:
        paths[key] = rel(copy_image(acaf_charts / name, f"assets/images/acaf/{key}.png", 1300, 880))
    paths["acaf_arch"] = rel(copy_static(acaf_charts / "architecture_dataflow.svg", "assets/images/acaf/architecture.svg"))

    fitness = SOURCE_ROOT / "AI健身项目" / "portfolio_pack" / "素材包"
    for name, key in [
        ("深蹲-展示.png", "fitness_squat"),
        ("俯卧撑-展示.png", "fitness_pushup"),
        ("引体向上-展示.png", "fitness_pullup"),
        ("仰卧起坐-展示.png", "fitness_situp"),
    ]:
        paths[key] = rel(copy_image(fitness / "截图" / name, f"assets/images/fitness/{key}.jpg", 920, 1080))
    paths["fitness_strip"] = rel(make_media_strip(
        [fitness / "截图" / n for n in ["深蹲-展示.png", "俯卧撑-展示.png", "引体向上-展示.png", "仰卧起坐-展示.png"]],
        "assets/images/fitness/fitness-strip.jpg",
        ["深蹲", "俯卧撑", "引体向上", "仰卧起坐"],
    ))
    paths["fitness_bg"] = rel(draw_blue_tech_background("assets/images/fitness/blue-tech-hero.jpg"))
    for video_name, key in [
        ("out-深蹲-左侧面.mp4", "fitness_gif_squat"),
        ("out-宽距俯卧撑-正面.mp4", "fitness_gif_pushup"),
        ("引体向上-背面.mp4", "fitness_gif_pullup"),
        ("out-仰卧起坐-右侧面.mp4", "fitness_gif_situp"),
    ]:
        paths[key] = rel(make_video_gif(fitness / "视频" / video_name, f"assets/media/fitness/{key}.gif", start_sec=1.0, duration=2.8, fps_out=6, width=340, max_frames=18))

    quant_pkg = SOURCE_ROOT / "量化系统正式版V1" / "portfolio_pack"
    quant_assets = quant_pkg / "素材包"
    paths["quant_latest"] = rel(copy_image(quant_assets / "最新预测.png", "assets/images/quant/latest-prediction.jpg", 1400, 900))
    paths["quant_sim"] = rel(copy_image(quant_assets / "真实模拟.png", "assets/images/quant/real-simulation.jpg", 1400, 900))
    quant_chart_dir = quant_pkg / "assets" / "charts"
    paths["quant_old_top3"] = rel(copy_image(quant_chart_dir / "latest_prediction_top3.png", "assets/images/quant/latest-prediction-top3-source.png", 1300, 900))
    paths["quant_old_auc"] = rel(copy_image(quant_chart_dir / "test_auc_by_model.png", "assets/images/quant/test-auc-source.png", 1300, 900))
    eval_chart, top_chart = draw_quant_charts(
        quant_pkg / "assets" / "data" / "latest_model_evaluation.csv",
        quant_pkg / "assets" / "data" / "latest_model_top3.csv",
    )
    paths["quant_eval"] = rel(eval_chart)
    paths["quant_top3"] = rel(top_chart)
    paths["quant_bg"] = rel(draw_quant_agent_background("assets/images/quant/agent-candlestick-hero.jpg"))

    agent_pack = SOURCE_ROOT / "事件新闻情绪感知agent" / "portfolio_pack"
    paths["agent_arch"] = rel(copy_static(agent_pack / "assets" / "architecture_flow.svg", "assets/images/quant/event-agent-architecture.svg"))
    paths["agent_card"] = rel(copy_static(agent_pack / "assets" / "resume_zip_pack" / "项目概览卡片.svg", "assets/images/quant/event-agent-card.svg"))

    rail = SOURCE_ROOT / "铁路部件拼接项目"
    rail_common = rail / "portfolio_pack" / "common_assets"
    rail_show = rail / "铁路部件识别项目（视频文件转图像拼接）" / "展示资料"
    paths["rail_final"] = rel(copy_image(rail_show / "拼接全景图.jpg", "assets/images/railway/final-panorama.jpg", 1900, 1200))
    paths["rail_frame1"] = rel(copy_image(rail_common / "frame_01.jpg", "assets/images/railway/frame-01.jpg", 1000, 700))
    paths["rail_frame29"] = rel(copy_image(rail_common / "frame_29.jpg", "assets/images/railway/frame-29.jpg", 1000, 700))
    paths["rail_frame58"] = rel(copy_image(rail_common / "frame_58.jpg", "assets/images/railway/frame-58.jpg", 1000, 700))
    paths["rail_video_gif"] = rel(make_video_gif(rail_show / "无人机视频.mp4", "assets/media/railway/uav-preview.gif", start_sec=2.0, duration=2.8, fps_out=6, width=520, max_frames=18))
    paths["rail_reveal"] = rel(make_reveal_gif(rail_show / "拼接全景图.jpg", "assets/media/railway/panorama-build.gif", width=820))

    logistics = SOURCE_ROOT / "物流大数据垂直大模型" / "portfolio_pack" / "web_showcase_assets" / "assets"
    for name, key in [
        ("architecture_diagram.png", "log_arch"),
        ("dashboard_overview.png", "log_dashboard"),
        ("route_optimization_demo.png", "log_route"),
        ("rag_skill_trace.png", "log_rag"),
    ]:
        paths[key] = rel(copy_image(logistics / name, f"assets/images/logistics/{key}.jpg", 1400, 900))
    paths["log_bg"] = rel(draw_logistics_background("assets/images/logistics/logistics-hero.jpg"))

    campus = SOURCE_ROOT / "校园网约车项目" / "portfolio_pack" / "素材包"
    shuy = SOURCE_ROOT / "舒腰健脊App开发" / "portfolio_pack" / "素材包"
    mountain = SOURCE_ROOT / "登山协会小程序" / "portfolio_pack" / "素材包"
    mini_srcs = [
        campus / "截图" / "01-首页_拼车调度台.png",
        shuy / "截图" / "患者端-首页.jpg",
        mountain / "截图" / "首页.jpg",
    ]
    paths["mini_collage"] = rel(make_miniapp_collage(mini_srcs, "assets/images/miniapps/miniapp-collage.jpg"))
    paths["mini_bg"] = rel(draw_miniapp_background("assets/images/miniapps/devtools-hero.jpg"))
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
            paths[key] = rel(copy_image(src, f"assets/images/miniapps/{key}.jpg", 720, 1120))
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
  --ink: #152027;
  --muted: #60707b;
  --line: #dce5ea;
  --paper: #f5f7f8;
  --panel: #ffffff;
  --blue: #135577;
  --teal: #1c827d;
  --gold: #bd8d28;
  --shadow: 0 18px 42px rgba(15, 28, 35, .10);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: #fbfcfc;
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif;
  line-height: 1.64;
}
a { color: inherit; text-decoration: none; }
img, video { display: block; max-width: 100%; }
.site-header {
  position: sticky;
  top: 0;
  z-index: 40;
  height: 58px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 0 5.5vw;
  background: rgba(255, 255, 255, .88);
  border-bottom: 1px solid rgba(218, 228, 234, .8);
  backdrop-filter: blur(16px);
}
.brand {
  font-weight: 900;
  color: #102231;
  white-space: nowrap;
}
.site-header nav {
  display: flex;
  gap: 17px;
  overflow-x: auto;
  white-space: nowrap;
  color: #315063;
  font-size: 14px;
  font-weight: 800;
}
.site-header nav a { padding: 18px 0; }
.immersive-hero {
  position: relative;
  min-height: calc(82vh - 58px);
  display: grid;
  align-items: end;
  overflow: hidden;
  color: #fff;
  background: #071015;
}
.immersive-hero .hero-bg,
.immersive-hero .hero-slides,
.immersive-hero .hero-slide {
  position: absolute;
  inset: 0;
}
.immersive-hero .hero-bg {
  background-image: var(--hero-image);
  background-size: cover;
  background-position: center;
  transform: scale(1.01);
}
.hero-contain .hero-bg {
  background-size: contain;
  background-repeat: no-repeat;
  background-color: #071015;
}
.hero-slide { opacity: 0; transition: opacity 900ms ease; margin: 0; }
.hero-slide.is-active { opacity: 1; }
.hero-slide img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center;
}
.hero-shade {
  position: absolute;
  inset: 0;
  background:
    linear-gradient(90deg, rgba(6, 13, 18, .88), rgba(6, 13, 18, .46) 48%, rgba(6, 13, 18, .18)),
    linear-gradient(0deg, rgba(6, 13, 18, .75), rgba(6, 13, 18, .12) 55%);
}
.hero-inner {
  position: relative;
  width: min(1180px, calc(100% - 48px));
  margin: 0 auto;
  padding: 0 0 68px;
}
.home-immersive { min-height: calc(88vh - 58px); }
.kicker {
  margin: 0 0 14px;
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 0;
  color: rgba(255, 255, 255, .84);
}
h1 {
  margin: 0;
  font-size: clamp(42px, 7.2vw, 94px);
  line-height: 1.05;
  letter-spacing: 0;
  font-weight: 950;
  overflow-wrap: anywhere;
  word-break: break-word;
}
.lead {
  max-width: 900px;
  margin: 22px 0 0;
  color: rgba(255, 255, 255, .92);
  font-size: clamp(18px, 2.1vw, 26px);
  line-height: 1.58;
}
.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 24px;
}
.tag-row span {
  color: #0d4d6d;
  background: rgba(245, 252, 255, .94);
  border: 1px solid rgba(214, 239, 248, .72);
  border-radius: 8px;
  padding: 7px 12px;
  font-size: 14px;
  font-weight: 900;
  box-shadow: 0 8px 22px rgba(0, 0, 0, .16);
}
.hero-dots {
  position: absolute;
  left: 50%;
  bottom: 26px;
  transform: translateX(-50%);
  display: flex;
  gap: 9px;
  z-index: 5;
}
.hero-dots button {
  width: 34px;
  height: 4px;
  border: 0;
  padding: 0;
  border-radius: 999px;
  background: rgba(255, 255, 255, .42);
}
.hero-dots button.is-active { background: #fff; }
.section {
  width: min(1180px, calc(100% - 44px));
  margin: 0 auto;
  padding: 74px 0 0;
}
.section-heading {
  display: grid;
  grid-template-columns: minmax(220px, .72fr) minmax(0, 1.28fr);
  gap: 34px;
  align-items: end;
  margin-bottom: 24px;
}
.section-heading h2 {
  margin: 0;
  font-size: clamp(28px, 3.2vw, 46px);
  line-height: 1.16;
  letter-spacing: 0;
}
.section-heading p {
  margin: 0;
  max-width: 760px;
  color: var(--muted);
  font-size: 17px;
}
.portfolio-stack {
  display: grid;
  gap: 28px;
}
.showcase-item {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(320px, .95fr);
  gap: 34px;
  align-items: stretch;
  min-height: 360px;
  padding: 18px;
  border: 1px solid rgba(210, 222, 230, .88);
  background:
    linear-gradient(135deg, rgba(255, 255, 255, .96), rgba(247, 251, 252, .94));
  box-shadow: var(--shadow);
}
.showcase-item:nth-child(even) .showcase-media { order: 2; }
.showcase-media {
  position: relative;
  min-height: 330px;
  background: #101820;
  overflow: hidden;
}
.showcase-media img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #101820;
}
.showcase-copy {
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 18px 18px 18px 4px;
}
.showcase-copy .num {
  color: var(--gold);
  font-weight: 950;
  letter-spacing: 0;
  margin-bottom: 10px;
}
.showcase-copy h3 {
  margin: 0;
  font-size: clamp(25px, 3vw, 38px);
  line-height: 1.18;
  letter-spacing: 0;
}
.showcase-copy p {
  margin: 14px 0 0;
  color: #53636e;
  font-size: 16px;
}
.link-button {
  width: max-content;
  margin-top: 22px;
  padding: 10px 15px;
  border: 1px solid #b9cbd5;
  border-radius: 8px;
  color: #174f73;
  background: #eef8fc;
  font-weight: 900;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
.metric {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 19px 20px;
  box-shadow: 0 10px 24px rgba(16, 31, 40, .05);
}
.metric strong {
  display: block;
  font-size: clamp(25px, 3vw, 38px);
  color: #125477;
  line-height: 1.08;
  overflow-wrap: anywhere;
  word-break: break-word;
}
.metric span {
  display: block;
  margin-top: 8px;
  color: var(--muted);
}
.gallery {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 20px;
}
.gallery.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.gallery.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.picture-panel {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 12px 28px rgba(19, 35, 43, .06);
}
.picture-panel .picture {
  min-height: 240px;
  display: grid;
  place-items: center;
  background: #eef2f4;
}
.picture-panel img,
.picture-panel video {
  width: 100%;
  height: 100%;
  max-height: 520px;
  object-fit: contain;
}
.picture-panel.tall .picture { min-height: 420px; }
.picture-copy {
  padding: 15px 16px 17px;
}
.picture-copy h3,
.text-panel h3 {
  margin: 0 0 7px;
  font-size: 18px;
  letter-spacing: 0;
}
.picture-copy p,
.text-panel p,
.note,
.rich-copy p,
.rich-copy li {
  color: var(--muted);
  margin: 0;
}
.wide-media {
  background: #0e171d;
  border: 1px solid var(--line);
  padding: 14px;
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.wide-media img,
.wide-media video {
  width: 100%;
  height: auto;
  object-fit: contain;
  background: #0e171d;
}
.sequence-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}
.sequence-step {
  position: relative;
  min-height: 170px;
  padding: 20px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, .98), rgba(245, 250, 251, .96));
  box-shadow: 0 12px 30px rgba(16, 31, 40, .06);
  overflow: hidden;
}
.sequence-step::after {
  content: "";
  position: absolute;
  inset: auto -20px -38px auto;
  width: 132px;
  height: 132px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(28, 130, 125, .16), transparent 68%);
}
.sequence-step span {
  display: inline-flex;
  color: var(--gold);
  font-weight: 950;
}
.sequence-step h3 {
  margin: 8px 0 8px;
  font-size: 21px;
  letter-spacing: 0;
}
.sequence-step p {
  margin: 0;
  color: var(--muted);
}
.workbench-stage {
  display: grid;
  grid-template-columns: minmax(0, 1.46fr) minmax(320px, .54fr);
  gap: 22px;
  align-items: stretch;
}
.workbench-stage.reversed {
  grid-template-columns: minmax(320px, .62fr) minmax(0, 1.38fr);
}
.stage-media {
  min-height: 520px;
  display: grid;
  align-items: center;
  padding: 18px;
}
.stage-media img,
.stage-media video {
  max-height: none;
}
.stage-copy {
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.stage-copy .eyebrow {
  margin: 0 0 8px;
  color: var(--gold);
  font-weight: 950;
}
.subsection-label {
  margin: 28px 0 14px;
  color: #102231;
  font-size: 24px;
  font-weight: 950;
  letter-spacing: 0;
}
.evidence-layout {
  display: grid;
  grid-template-columns: minmax(320px, .85fr) minmax(0, 1.15fr);
  gap: 20px;
  align-items: start;
}
.evidence-layout .sector-board {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.risk-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}
.agent-summary {
  display: grid;
  grid-template-columns: minmax(0, .88fr) minmax(0, 1.12fr);
  gap: 20px;
  align-items: start;
}
.agent-summary .metrics {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.agent-summary .metric strong {
  font-size: clamp(24px, 2.4vw, 34px);
}
.process {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
.process div,
.text-panel,
.note {
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 8px;
  padding: 18px;
  box-shadow: 0 10px 24px rgba(16, 31, 40, .05);
}
.process strong {
  display: block;
  margin-bottom: 8px;
  color: #112b3a;
  font-size: 17px;
}
.process span {
  color: var(--muted);
}
.text-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}
.fusion-map {
  display: grid;
  grid-template-columns: minmax(260px, .95fr) minmax(0, 1.35fr) minmax(260px, .95fr);
  gap: 16px;
  align-items: stretch;
}
.fusion-panel {
  position: relative;
  min-height: 270px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 22px;
  overflow: hidden;
  background:
    linear-gradient(145deg, rgba(8, 23, 30, .96), rgba(17, 54, 60, .92));
  color: #fff;
  box-shadow: var(--shadow);
}
.fusion-panel h3,
.event-card h3,
.etf-card h3,
.risk-card h3 {
  margin: 0 0 10px;
  letter-spacing: 0;
}
.fusion-panel p {
  margin: 0;
  color: rgba(236, 249, 248, .78);
}
.fusion-panel ul {
  margin: 16px 0 0;
  padding-left: 19px;
  color: rgba(236, 249, 248, .82);
}
.fusion-center {
  display: grid;
  place-items: center;
  text-align: center;
  background:
    radial-gradient(circle at center, rgba(56, 189, 170, .28), transparent 48%),
    linear-gradient(145deg, rgba(7, 20, 27, .98), rgba(14, 42, 50, .96));
}
.fusion-center strong {
  display: block;
  font-size: clamp(32px, 5vw, 64px);
  line-height: 1;
}
.fusion-center span {
  color: rgba(236, 249, 248, .78);
  font-weight: 800;
}
.channel-grid {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 12px;
}
.channel-card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  min-height: 128px;
  box-shadow: 0 10px 24px rgba(16, 31, 40, .05);
}
.channel-card strong {
  display: block;
  color: #115571;
  font-size: 26px;
  line-height: 1;
}
.channel-card span {
  display: block;
  margin-top: 9px;
  color: var(--muted);
  font-size: 14px;
}
.sector-board {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
.sector-card {
  position: relative;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 10px 24px rgba(16, 31, 40, .05);
  overflow: hidden;
}
.sector-card::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 5px;
  background: linear-gradient(180deg, #1c827d, #bd8d28);
}
.sector-card h3 {
  margin: 0;
  font-size: 19px;
}
.sector-score {
  display: block;
  margin-top: 12px;
  color: #0e5574;
  font-size: 34px;
  font-weight: 950;
  line-height: 1;
}
.sector-card p {
  margin: 10px 0 0;
  color: var(--muted);
}
.etf-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.etf-card,
.risk-card,
.event-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  padding: 18px;
  box-shadow: 0 10px 24px rgba(16, 31, 40, .05);
}
.etf-card .code,
.risk-card .code {
  display: inline-flex;
  margin-bottom: 10px;
  padding: 5px 8px;
  color: #0e5574;
  background: #eef8fc;
  border: 1px solid #cbe2ec;
  border-radius: 7px;
  font-weight: 950;
}
.score-line {
  display: grid;
  grid-template-columns: 90px 1fr 58px;
  gap: 10px;
  align-items: center;
  margin: 10px 0;
  color: var(--muted);
  font-size: 14px;
}
.score-track {
  height: 9px;
  border-radius: 999px;
  background: #e7eef2;
  overflow: hidden;
}
.score-track i {
  display: block;
  height: 100%;
  width: var(--score-width, 50%);
  border-radius: inherit;
  background: linear-gradient(90deg, #1c827d, #49a98d);
}
.risk-card .score-track i {
  background: linear-gradient(90deg, #b95c49, #d59a43);
}
.reason-list {
  margin: 12px 0 0;
  padding-left: 18px;
  color: var(--muted);
}
.event-timeline {
  display: grid;
  gap: 13px;
}
.event-card {
  display: grid;
  grid-template-columns: 140px minmax(0, 1fr) 100px;
  gap: 16px;
  align-items: center;
}
.event-card .event-meta {
  color: #0e5574;
  font-size: 14px;
  font-weight: 950;
}
.event-card p {
  margin: 0;
  color: #40515c;
}
.event-card .impact {
  justify-self: end;
  font-weight: 950;
  color: #1c827d;
}
.event-card.negative .impact { color: #b95c49; }
.notice-strip {
  border: 1px solid #e3d7b7;
  border-left: 5px solid var(--gold);
  border-radius: 8px;
  padding: 18px 20px;
  background: #fffaf0;
  color: #5f5030;
  box-shadow: 0 10px 24px rgba(16, 31, 40, .04);
}
.feature-split {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(300px, .9fr);
  gap: 22px;
  align-items: stretch;
}
.rich-copy {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  padding: 22px;
  box-shadow: 0 10px 24px rgba(16, 31, 40, .05);
}
.rich-copy h3 {
  margin: 0 0 10px;
  font-size: 22px;
}
.rich-copy ul {
  margin: 12px 0 0;
  padding-left: 20px;
}
.phone-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}
.phone-card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 10px 24px rgba(16, 31, 40, .05);
}
.phone-card img {
  width: 100%;
  height: 500px;
  object-fit: contain;
  background: #f3f7f8;
}
.phone-card p {
  margin: 10px 0 2px;
  font-weight: 900;
  color: #1c3544;
}
.site-footer {
  width: min(1180px, calc(100% - 44px));
  margin: 70px auto 0;
  padding: 26px 0 42px;
  color: #6b7880;
  border-top: 1px solid var(--line);
  font-size: 14px;
}
@media (max-width: 980px) {
  .site-header { padding: 0 18px; }
  .brand { display: none; }
  .site-header nav { width: 100%; }
  .section-heading,
  .showcase-item,
  .feature-split,
  .gallery,
  .gallery.three,
  .gallery.four,
  .text-grid,
  .process,
  .metrics,
  .sequence-strip,
  .workbench-stage,
  .workbench-stage.reversed,
  .evidence-layout,
  .risk-grid,
  .agent-summary,
  .agent-summary .metrics {
    grid-template-columns: 1fr;
  }
  .showcase-item:nth-child(even) .showcase-media { order: 0; }
  .showcase-media { min-height: 260px; }
  .stage-media { min-height: 320px; }
  .phone-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .phone-card img { height: 420px; }
  .fusion-map,
  .channel-grid,
  .sector-board,
  .etf-grid,
  .event-card {
    grid-template-columns: 1fr;
  }
  .event-card .impact { justify-self: start; }
}
@media (max-width: 620px) {
  .immersive-hero { min-height: calc(76vh - 58px); }
  .hero-inner { width: min(100% - 28px, 1180px); padding-bottom: 52px; }
  h1 {
    font-size: clamp(30px, 8.8vw, 44px);
    line-height: 1.1;
    word-break: break-all;
  }
  .lead { font-size: 16px; }
  .section { width: min(100% - 28px, 1180px); padding-top: 52px; }
  .showcase-item { padding: 12px; gap: 14px; }
  .phone-grid { grid-template-columns: 1fr; }
}
"""
    js = r"""
(function () {
  function runCarousel(root, slideSelector, dotSelector, interval) {
    var slides = Array.prototype.slice.call(root.querySelectorAll(slideSelector));
    var dots = Array.prototype.slice.call(root.querySelectorAll(dotSelector));
    if (slides.length < 2) return;
    var index = 0;
    function show(next) {
      slides[index].classList.remove("is-active");
      if (dots[index]) dots[index].classList.remove("is-active");
      index = (next + slides.length) % slides.length;
      slides[index].classList.add("is-active");
      if (dots[index]) dots[index].classList.add("is-active");
    }
    dots.forEach(function (dot, i) {
      dot.addEventListener("click", function () { show(i); });
    });
    window.setInterval(function () { show(index + 1); }, interval);
  }
  document.querySelectorAll("[data-hero-carousel]").forEach(function (root) {
    runCarousel(root, ".hero-slide", ".hero-dots button", 4200);
  });
})();
"""
    (ASSETS / "styles.css").write_text(css, encoding="utf-8")
    (ASSETS / "site.js").write_text(js, encoding="utf-8")


def write_index(paths: dict[str, str]) -> None:
    cards = [
        ("01", "360车载全景影像拼接", "四路鱼眼相机输入、BEV 几何映射、区域融合与上机实测，补充低延迟视频流处理升级。", paths["avm_demo_bev"], "projects/avm-360.html", ["OpenCV", "Fisheye", "BEV"]),
        ("02", "AFAC2025金融智能创新大赛", "20 只基金未来 7 天申购与赎回预测，多源特征工程与 LSTM-Attention 双通道建模。", paths["acaf_award"], "projects/acaf-finance.html", ["PyTorch", "时间序列", "金融智能"]),
        ("03", "AI健身动作识别与评分系统", "摄像头姿态估计、关节角度状态机、动作计数、评分与训练反馈展示。", paths["fitness_strip"], "projects/ai-fitness.html", ["MediaPipe", "OpenCV", "运动健康"]),
        ("04", "事件与情绪分析智能体 / ETF量化辅助系统", "ETF 横截面轮动主引擎结合事件新闻情绪层，用于候选排序、解释和模拟调仓。", paths["quant_sim"], "projects/quant-agent.html", ["ETF轮动", "Agent", "模拟决策"]),
        ("05", "铁路无人机视频全景拼接", "从连续无人机视频帧生成铁路场景全景底图，支撑巡检场景的大视野观察。", paths["rail_final"], "projects/railway-stitching.html", ["SIFT", "RANSAC", "全景拼接"]),
        ("06", "物流大数据垂直大模型", "RAG 知识库、物流领域工具调用、路径优化、风险预警与自动周报工作台。", paths["log_dashboard"], "projects/logistics-llm.html", ["RAG", "Streamlit", "路径优化"]),
        ("07", "微信小程序展示合集", "校园网约车、舒腰健脊、登山协会三个小程序，展示产品流程与前端实现能力。", paths["mini_collage"], "projects/miniapps.html", ["微信小程序", "Vue", "产品原型"]),
    ]
    card_html = ""
    for num, title, desc, img, href, tags in cards:
        card_html += f"""
        <article class="showcase-item">
          <a class="showcase-media" href="{esc(href)}"><img src="{esc(img)}" alt="{esc(title)}" loading="lazy"></a>
          <div class="showcase-copy">
            <div class="num">{esc(num)}</div>
            <h3>{esc(title)}</h3>
            <p>{esc(desc)}</p>
            <div class="tag-row">{tags_html(tags)}</div>
            <a class="link-button" href="{esc(href)}">查看项目</a>
          </div>
        </article>
        """
    body = home_hero([
        paths["acaf_award"],
        paths["acaf_trophy"],
        paths["acaf_roadshow"],
        paths["rail_final"],
        paths["quant_latest"],
    ])
    body += section("代表项目", f'<div class="portfolio-stack">{card_html}</div>', subtitle="以项目目标、关键做法和可视化成果为主线组织，首页保留静态展示，详细页展开完整材料。")
    (REPO / "index.html").write_text(page_shell("项目作品集概览", "计算机视觉、计算成像、金融智能、RAG 与微信小程序项目作品集", body), encoding="utf-8")


def write_avm(paths: dict[str, str]) -> None:
    body = immersive_hero(
        "360车载全景影像拼接",
        "工程车辆四路鱼眼相机 · BEV 环视拼接",
        "基于前、后、左、右四路鱼眼相机图像，完成去畸变、地面单应性投影、区域融合、上机展示和低延迟处理升级。",
        f"../{paths['avm_raw_front']}",
        ["OpenCV", "Fisheye", "Homography", "BEV Fusion"],
        contain=True,
    )
    body += section("相关参数", metric_cards([
        ("输入相机", "4 路"),
        ("标定点", "每路 8 个外角点"),
        ("back 重投影误差", "1.352px"),
        ("优化后处理均值", "95.9ms"),
    ]))
    body += section("输入图像", f"""
      <div class="gallery four">
        {picture_panel(f"../{paths['avm_raw_front']}", "前向原始图像")}
        {picture_panel(f"../{paths['avm_raw_left']}", "左侧原始图像")}
        {picture_panel(f"../{paths['avm_raw_right']}", "右侧原始图像")}
        {picture_panel(f"../{paths['avm_raw_back']}", "后向原始图像")}
      </div>
    """, subtitle="四路鱼眼图像作为 BEV 拼接输入，页面采用等比缩放展示，避免裁掉有效信息。")
    body += section("结果展示", f"""
      <div class="gallery">
        {picture_panel(f"../{paths['avm_baseline']}", "基础 BEV 融合结果")}
        {picture_panel(f"../{paths['avm_owned']}", "ownership-prior 区域融合")}
        {picture_panel(f"../{paths['avm_safe']}", "extended safe 输出")}
        {picture_panel(f"../{paths['avm_wide']}", "extended wide 输出")}
      </div>
    """)
    body += section("上机实测", f"""
      <div class="gallery three">
        {picture_panel(f"../{paths['avm_demo_front']}", "前向上机画面")}
        {picture_panel(f"../{paths['avm_demo_left']}", "左侧上机画面")}
        {picture_panel(f"../{paths['avm_demo_right']}", "右侧上机画面")}
        {picture_panel(f"../{paths['avm_demo_back']}", "后向上机画面")}
        {picture_panel(f"../{paths['avm_demo_bev']}", "上机鸟瞰输出")}
      </div>
    """)
    latency_images = ""
    if "avm_latency_compare" in paths:
        latency_images += picture_panel(f"../{paths['avm_latency_compare']}", "优化前后对比")
    if "avm_latency_breakdown" in paths:
        latency_images += picture_panel(f"../{paths['avm_latency_breakdown']}", "处理耗时分解")
    body += section("视频流低延迟升级", f"""
      <div class="feature-split">
        <div class="rich-copy">
          <h3>从离线拼接走向视频流处理</h3>
          <p>在原始拼接链路上补充四路 UDP 视频流输入与运行接口，并围绕 warp、bottom multiH、blend 和内存分配环节做处理耗时优化。</p>
          <ul>
            <li>原始处理均值约 1025.1ms，输出约 0.98 FPS。</li>
            <li>优化后处理均值约 95.9ms，输出约 10.43 FPS。</li>
            <li>四路端口接口按 front / left / right / bottom 分离，便于接入多摄像头视频流。</li>
          </ul>
        </div>
        <div class="gallery">{latency_images}</div>
      </div>
    """)
    body += section("算法流程", """
      <div class="process">
        <div><strong>相机输入</strong><span>采集四路鱼眼图像并统一标定点配置。</span></div>
        <div><strong>几何映射</strong><span>去畸变后通过地面单应性投影到 BEV 坐标系。</span></div>
        <div><strong>区域融合</strong><span>结合有效区域 mask、距离权重与 ownership-prior 降低重影。</span></div>
        <div><strong>展示输出</strong><span>生成多尺度 BEV 结果，并接入上机和视频流演示链路。</span></div>
      </div>
    """)
    (PROJECTS_DIR / "avm-360.html").write_text(page_shell("360车载全景影像拼接", "装载机四路鱼眼相机 BEV 拼接项目", body, active="avm"), encoding="utf-8")


def write_acaf(paths: dict[str, str]) -> None:
    body = immersive_hero(
        "基金产品长周期申购与赎回预测",
        "AFAC2025 金融智能创新大赛 · 晨曦组",
        "多源特征工程、LSTM-Attention 双通道建模，面向 20 只基金输出未来 7 天申购与赎回预测，并完成竞赛路演交付。",
        f"../{paths['acaf_roadshow']}",
        ["Python", "PyTorch", "时间序列预测", "金融数据建模", "特征工程"],
    )
    body += section("赛程数据", metric_cards([
        ("最终训练集", "9,460 行"),
        ("覆盖基金", "20 只"),
        ("训练字段", "18 列"),
        ("预测结果", "140 条"),
    ]))
    body += section("架构与数据流", f"""
      <div class="feature-split">
        <div class="wide-media">{media_tag(f"../{paths['acaf_arch']}", "项目架构与数据流图")}</div>
        <div class="rich-copy">
          <h3>从赛题数据到预测交付</h3>
          <p>项目将基金申赎曝光、百度指数、市场指数、节假日、发薪日概率与情绪特征对齐到基金-日期维度，再用 7 天历史窗口预测未来 7 天资金流。</p>
          <ul>
            <li>申购与赎回采用双通道建模，降低两个目标之间的信号混淆。</li>
            <li>注意力层用于提取时间窗口内的重点波动和节奏变化。</li>
            <li>输出结果转化为图表和路演材料，支撑现场讲解。</li>
          </ul>
        </div>
      </div>
    """)
    body += section("结果图表", f"""
      <div class="gallery">
        {picture_panel(f"../{paths['acaf_pred']}", "7 日申购/赎回预测汇总")}
        {picture_panel(f"../{paths['acaf_top10']}", "基金预测结果 Top10")}
        {picture_panel(f"../{paths['acaf_monthly']}", "训练数据月度变化")}
        {picture_panel(f"../{paths['acaf_corr']}", "主要特征相关性")}
      </div>
    """)
    body += section("路演与证书", f"""
      <div class="gallery three">
        {picture_panel(f"../{paths['acaf_award']}", "颁奖仪式")}
        {picture_panel(f"../{paths['acaf_trophy']}", "奖金奖杯")}
        {picture_panel(f"../{paths['acaf_certificate']}", "获奖证书")}
      </div>
    """)
    (PROJECTS_DIR / "acaf-finance.html").write_text(page_shell("AFAC2025金融智能创新大赛", "基金申赎 7 日预测项目", body, active="acaf"), encoding="utf-8")


def write_fitness(paths: dict[str, str]) -> None:
    body = immersive_hero(
        "AI健身动作识别与评分系统",
        "实时姿态估计 · 动作计数 · 运动反馈",
        "基于摄像头或视频输入进行人体关键点识别、关节角度计算、动作阶段判断、自动计数、评分和反馈展示。",
        f"../{paths['fitness_bg']}",
        ["MediaPipe Pose", "OpenCV", "状态机计数", "运动健康"],
    )
    body += section("核心流程", """
      <div class="process">
        <div><strong>视频输入</strong><span>摄像头或本地 MP4 由 OpenCV 逐帧读取。</span></div>
        <div><strong>姿态估计</strong><span>提取肩、肘、腕、髋、膝、踝等人体关键点。</span></div>
        <div><strong>规则判别</strong><span>通过关节角度和 up/down 阶段状态完成动作计数。</span></div>
        <div><strong>反馈展示</strong><span>将骨架、分数、次数、平均分和即时反馈叠加到画面。</span></div>
      </div>
    """)
    body += section("动作识别动图", f"""
      <div class="gallery four">
        {picture_panel(f"../{paths['fitness_gif_squat']}", "深蹲识别", "膝关节与髋关节角度用于阶段判断。")}
        {picture_panel(f"../{paths['fitness_gif_pushup']}", "俯卧撑识别", "肩肘腕关键点用于计数和评分反馈。")}
        {picture_panel(f"../{paths['fitness_gif_pullup']}", "引体向上识别", "背面姿态画面中识别肩臂动作变化。")}
        {picture_panel(f"../{paths['fitness_gif_situp']}", "仰卧起坐识别", "肩髋膝角度用于起身与躺下阶段判断。")}
      </div>
    """, subtitle="四类动作统一按等比缩放展示，保留视频画面中的骨架、计数和反馈信息。")
    body += section("项目截图", f"""
      <div class="gallery four">
        {picture_panel(f"../{paths['fitness_squat']}", "深蹲截图")}
        {picture_panel(f"../{paths['fitness_pushup']}", "俯卧撑截图")}
        {picture_panel(f"../{paths['fitness_pullup']}", "引体向上截图")}
        {picture_panel(f"../{paths['fitness_situp']}", "仰卧起坐截图")}
      </div>
    """)
    body += section("能力拆解", """
      <div class="text-grid">
        <article class="text-panel"><h3>视觉感知</h3><p>把实时视频画面转化为人体关键点序列，为动作判断提供稳定输入。</p></article>
        <article class="text-panel"><h3>规则建模</h3><p>不同动作对应不同角度组合与阶段阈值，避免单帧误触发导致重复计数。</p></article>
        <article class="text-panel"><h3>产品展示</h3><p>将分数、次数、热量和动作反馈组织到前端展示链路，形成可演示闭环。</p></article>
      </div>
    """)
    (PROJECTS_DIR / "ai-fitness.html").write_text(page_shell("AI健身动作识别与评分系统", "姿态估计、动作计数与运动反馈项目", body, active="fitness"), encoding="utf-8")


def write_quant(paths: dict[str, str]) -> None:
    agent_pack = SOURCE_ROOT / "事件新闻情绪感知agent" / "portfolio_pack"
    summary = json.loads((agent_pack / "data" / "portfolio_summary.json").read_text(encoding="utf-8"))
    top_etfs = read_csv_dicts(agent_pack / "data" / "top_etf_overlays_v2.csv")
    risk_etfs = read_csv_dicts(agent_pack / "data" / "risk_etf_overlays_v2.csv")
    sectors = read_csv_dicts(agent_pack / "data" / "top_sectors_v2.csv")
    events = read_csv_dicts(agent_pack / "data" / "representative_events_v2.csv")

    channel_names = {
        "media": "媒体叙事",
        "disclosure": "专业披露",
        "industry": "产业催化",
        "policy": "政策监管",
        "macro": "宏观经济",
        "community": "社区情绪",
        "geopolitical": "地缘风险",
    }
    channel_html = ""
    for key in ["media", "disclosure", "industry", "policy", "macro", "community", "geopolitical"]:
        channel_html += f"""
        <article class="channel-card">
          <strong>{esc(str(summary['impact_channel_counts'].get(key, 0)))}</strong>
          <span>{esc(channel_names[key])}</span>
        </article>
        """

    sector_html = ""
    for item in sectors[:4]:
        sector_html += f"""
        <article class="sector-card">
          <h3>{esc(item['板块桶'])}</h3>
          <span class="sector-score">{esc(safe_float(item['日度分'], 3))}</span>
          <p>{esc(item['状态'])} · 活跃事件 {esc(item['活跃事件数'])} 个</p>
          <p>媒体 {esc(safe_float(item['媒体叙事'], 3))} / 产业 {esc(safe_float(item['产业催化'], 3))} / 政策 {esc(safe_float(item['政策支持'], 3))}</p>
        </article>
        """

    def etf_card(item: dict[str, str]) -> str:
        total = float(item["总信息分"])
        risk = float(item["负向风险分"])
        total_w = max(4, min(100, abs(total) / 3.5 * 100))
        risk_w = max(4, min(100, risk / 0.37 * 100))
        reasons = "".join(f"<li>{esc(reason)}</li>" for reason in reason_list(item.get("今日变化原因", "")))
        return f"""
        <article class="etf-card">
          <span class="code">{esc(item['代码'])}</span>
          <h3>{esc(item['名称'])}</h3>
          <div class="score-line"><span>信息分</span><div class="score-track"><i style="--score-width:{total_w:.1f}%"></i></div><strong>{esc(safe_float(total, 3))}</strong></div>
          <div class="score-line"><span>风险分</span><div class="score-track"><i style="--score-width:{risk_w:.1f}%"></i></div><strong>{esc(safe_float(risk, 3))}</strong></div>
          <p>{esc(item['板块桶'])} · {esc(item['标签'])} · {esc(item['覆盖状态'])}</p>
          <ul class="reason-list">{reasons}</ul>
        </article>
        """

    etf_html = "".join(etf_card(item) for item in top_etfs[:4])

    def risk_card(item: dict[str, str]) -> str:
        risk = float(item["负向风险分"])
        risk_w = max(4, min(100, risk / 0.37 * 100))
        return f"""
        <article class="risk-card">
          <span class="code">{esc(item['代码'])}</span>
          <h3>{esc(item['名称'])}</h3>
          <div class="score-line"><span>负向风险</span><div class="score-track"><i style="--score-width:{risk_w:.1f}%"></i></div><strong>{esc(safe_float(risk, 3))}</strong></div>
          <p>{esc(item['板块桶'])} · {esc(item['标签'])} · 信息分 {esc(safe_float(item['总信息分'], 3))}</p>
        </article>
        """

    risk_html = "".join(risk_card(item) for item in risk_etfs[:4])

    event_html = ""
    for item in events[:6]:
        impact = float(item["影响分"])
        cls = " negative" if impact < 0 else ""
        event_html += f"""
        <article class="event-card{cls}">
          <div class="event-meta">{esc(item['事件大类'])}<br>{esc(item['通道'])}</div>
          <p>{esc(item['摘要'])}</p>
          <div class="impact">{esc(safe_float(impact, 3))}<br><span>{esc(safe_float(item['置信度'], 2))}</span></div>
        </article>
        """

    body = immersive_hero(
        "事件与情绪分析智能体 / ETF量化辅助系统",
        "ETF 横截面轮动 · 事件新闻情绪 Agent · 模拟决策工作台",
        "页面按真实使用顺序展示：先由 ETF 量化工作台生成候选排序和模拟调仓，再由事件新闻情绪 Agent 读取公开信息，补充板块情绪、ETF 信息画像和风险复核线索。",
        f"../{paths['quant_bg']}",
        ["ETF轮动", "多模型候选排序", "事件情绪Agent", "信息增强画像", "风险复核"],
    )
    body += section("项目阅读路径", """
      <div class="sequence-strip">
        <article class="sequence-step">
          <span>01 ETF 主系统</span>
          <h3>先生成候选排序</h3>
          <p>基于 ETF 日线面板和固定因子，输出未来 10 个交易日的候选 ETF、概率与策略目标。</p>
        </article>
        <article class="sequence-step">
          <span>02 模拟与评估</span>
          <h3>再观察策略表现</h3>
          <p>从指定观察日开始滚动调仓，结合 Top3 概率和模型评估图验证工作台输出。</p>
        </article>
        <article class="sequence-step">
          <span>03 情绪 Agent</span>
          <h3>最后补充解释和风险</h3>
          <p>把新闻、披露、政策和社区评论转成事件对象，连接到板块与 ETF 画像中。</p>
        </article>
      </div>
    """, subtitle="这里不是把两个项目简单并排，而是把量化候选、事件解释和风险复核放到同一条工作流里。")
    body += section("系统数据", metric_cards([
        ("ETF 因子记录", "62,835 行"),
        ("量化交易日", "1,781 个"),
        ("真实公开信息", f"{summary['counts']['raw_records']:,} 条"),
        ("V2事件对象", f"{summary['counts']['events_v2']:,} 个"),
        ("板块状态", f"{summary['counts']['sector_states_v2']} 个"),
        ("ETF信息画像", f"{summary['counts']['etf_overlays_v2']} 个"),
    ]))
    body += section("ETF 量化工作台", f"""
      <div class="workbench-stage">
        <div class="wide-media stage-media">{media_tag(f"../{paths['quant_latest']}", "ETF 量化工作台最新预测界面")}</div>
        <div class="rich-copy stage-copy">
          <p class="eyebrow">第一层：候选排序</p>
          <h3>先回答“哪些 ETF 值得进入观察池”</h3>
          <p>工作台以 ETF 日线面板为基础，结合 46 个固定因子和多模型输出，形成横截面候选排序。界面保留预测日期、训练区间、模型结果、概率和策略目标，方便从候选池进入后续模拟。</p>
          <ul>
            <li>预测未来 10 个交易日相对候选池中位数的强弱。</li>
            <li>输出候选 ETF、模型概率、主题方向和可观察排序。</li>
            <li>作为后续事件解释的入口，而不是孤立展示一张预测表。</li>
          </ul>
        </div>
      </div>
    """)
    body += section("预测与模拟", f"""
      <div class="workbench-stage reversed">
        <div class="rich-copy stage-copy">
          <p class="eyebrow">第二层：模拟验证</p>
          <h3>把模型候选放进滚动调仓场景</h3>
          <p>从一个观察日开始，按所选策略每 10 个交易日滚动调仓，观察模拟资金曲线变化。这个部分展示的是量化主系统本身的输出能力，为事件 Agent 的解释层提供明确承接对象。</p>
          <ul>
            <li>真实模拟图展示策略滚动后的资金变化。</li>
            <li>Top3 概率图用于检查候选排序的集中度。</li>
            <li>模型评估图用于对比不同模型在测试集上的表现。</li>
          </ul>
        </div>
        <div class="wide-media stage-media">{media_tag(f"../{paths['quant_sim']}", "ETF 真实模拟调仓曲线")}</div>
      </div>
      <div class="gallery three" style="margin-top:22px">
        {picture_panel(f"../{paths['quant_top3']}", "最新 Top3 概率")}
        {picture_panel(f"../{paths['quant_eval']}", "模型评估示例")}
        {picture_panel(f"../{paths['quant_old_auc']}", "测试集 AUC 源图")}
      </div>
    """)
    body += section("事件新闻情绪 Agent", f"""
      <div class="agent-summary">
        <div>
          {metric_cards([
              ("运行批次", str(summary["batch_id"])),
              ("运行日期", str(summary["run_date"])),
              ("非零影响事件", f"{summary['counts']['nonzero_events_v2']:,} 个"),
              ("正向/负向事件", f"{summary['counts']['positive_events_v2']} / {summary['counts']['negative_events_v2']}"),
              ("平均置信度", safe_float(summary["counts"]["avg_event_confidence"], 4)),
              ("公开信息源", f"{summary['source_module_counts']['财经新闻'] + summary['source_module_counts']['专业披露与会议纪要'] + summary['source_module_counts']['基金社区评论']:,} 条"),
          ])}
        </div>
        <div class="rich-copy">
          <h3>把公开文本转成可接入量化结果的信息层</h3>
          <p>Agent 读取财经新闻、专业披露与会议纪要、基金社区评论等公开信息，经过获取、清洗、去重聚类、事件结构化和统一证据对象，生成可追溯的板块情绪与 ETF 信息增强画像。</p>
          <ul>
            <li>事件本体统一映射政策、宏观、产业、披露、地缘、媒体和社区情绪。</li>
            <li>规则主链路稳定生成结构化结果，LLM 作为可选增强用于摘要和归因。</li>
            <li>输出目标是解释量化候选、确认催化和提示需要复核的风险。</li>
          </ul>
        </div>
      </div>
      <div class="wide-media stage-media" style="margin-top:22px">{media_tag(f"../{paths['agent_arch']}", "事件新闻情绪 Agent 架构")}</div>
    """, subtitle="这一层放在量化结果之后出现：它不替代模型排序，而是解释排序背后的当日信息结构。")
    body += section("事件来源与影响通道", f"""
      <div class="metrics">
        <div class="metric"><strong>{summary['source_module_counts']['财经新闻']:,} 条</strong><span>财经新闻</span></div>
        <div class="metric"><strong>{summary['source_module_counts']['专业披露与会议纪要']:,} 条</strong><span>专业披露与会议纪要</span></div>
        <div class="metric"><strong>{summary['source_module_counts']['基金社区评论']:,} 条</strong><span>基金社区评论</span></div>
        <div class="metric"><strong>{summary['counts']['raw_records']:,} 条</strong><span>公开信息总量</span></div>
      </div>
      <h3 class="subsection-label">影响通道分布</h3>
      <div class="channel-grid">{channel_html}</div>
    """, subtitle="事件对象按影响通道进入后续聚合，媒体叙事提供高频线索，披露、政策和产业事件用于增强置信与解释。")
    body += section("情绪到画像", f"""
      <div class="evidence-layout">
        <div class="rich-copy">
          <h3>先聚合到板块，再映射到候选 ETF</h3>
          <p>Agent 不直接把单条新闻贴到 ETF 上，而是先形成板块状态，再根据主题映射关系把信息分、风险分和当日变化原因写入 ETF 画像。这样量化工作台的候选结果就有了可解释的信息侧视图。</p>
          <ul>
            <li>板块层：展示日度分、事件活跃度和主要信息来源。</li>
            <li>ETF 层：展示信息分、风险分、覆盖状态和代表性变化原因。</li>
            <li>证据层：可继续回溯到代表事件摘要、影响通道和置信度。</li>
          </ul>
        </div>
        <div class="sector-board">{sector_html}</div>
      </div>
      <h3 class="subsection-label">ETF 信息增强画像</h3>
      <div class="etf-grid">{etf_html}</div>
    """)
    body += section("风险与事件", f"""
      <div class="evidence-layout">
        <div>
          <h3 class="subsection-label" style="margin-top:0">风险复核清单</h3>
          <div class="risk-grid">{risk_html}</div>
        </div>
        <div>
          <h3 class="subsection-label" style="margin-top:0">代表事件链路</h3>
          <div class="event-timeline">{event_html}</div>
        </div>
      </div>
    """, subtitle="量化系统给出候选，事件 Agent 负责把支持因素和负向压力同时放到台面上，便于复盘和人工判断。")
    body += section("融合后的决策闭环", """
      <div class="fusion-map">
        <article class="fusion-panel">
          <h3>ETF 排序层</h3>
          <p>由历史因子、横截面标签和多模型结果生成候选池。</p>
          <ul>
            <li>回答“先看哪些 ETF”。</li>
            <li>沉淀为预测表、概率和模拟调仓记录。</li>
          </ul>
        </article>
        <article class="fusion-panel fusion-center">
          <div><strong>候选<br>画像<br>复核</strong><span>从模型输出走向可解释观察</span></div>
        </article>
        <article class="fusion-panel">
          <h3>事件解释层</h3>
          <p>把当日公开信息压缩成板块情绪、ETF 画像和风险线索。</p>
          <ul>
            <li>回答“为什么值得看”。</li>
            <li>提示“哪里需要谨慎”。</li>
          </ul>
        </article>
      </div>
      <div class="notice-strip" style="margin-top:16px">展示批次采用规则主链路稳定生成结构化结果，LLM 结构化作为可选增强；当外部模型不可用时，板块状态、ETF 信息画像和风险复核仍按规则链路输出。</div>
    """)
    (PROJECTS_DIR / "quant-agent.html").write_text(page_shell("事件与情绪分析智能体 / ETF量化辅助系统", "ETF 轮动、事件情绪和模拟决策系统", body, active="quant"), encoding="utf-8")


def write_railway(paths: dict[str, str]) -> None:
    body = immersive_hero(
        "铁路无人机视频全景拼接",
        "无人机影像 · 图像拼接 · OpenCV 原型",
        "面向铁路巡检视频视野碎片化的问题，完成视频抽帧、特征匹配、几何配准、全局位姿链和全景输出，形成可复盘的大视野铁路场景底图。",
        f"../{paths['rail_final']}",
        ["OpenCV", "SIFT / RootSIFT", "RANSAC", "无人机影像"],
        contain=True,
    )
    body += section("项目状态", metric_cards([
        ("输入帧", "58 张"),
        ("匹配边日志", "227 条"),
        ("位姿链", "58 帧"),
        ("全景输出", "13,567x11,902"),
    ]))
    body += section("视频与拼接过程", f"""
      <div class="gallery">
        {picture_panel(f"../{paths['rail_video_gif']}", "原始无人机视频片段", "连续航拍画面作为拼接输入。")}
        {picture_panel(f"../{paths['rail_reveal']}", "全景逐步成图", "用逐步展开的方式展示铁路场景底图的形成过程。")}
      </div>
    """)
    body += section("关键帧与全景结果", f"""
      <div class="gallery three">
        {picture_panel(f"../{paths['rail_frame1']}", "输入帧 01")}
        {picture_panel(f"../{paths['rail_frame29']}", "输入帧 29")}
        {picture_panel(f"../{paths['rail_frame58']}", "输入帧 58")}
      </div>
      <div class="wide-media" style="margin-top:20px">{media_tag(f"../{paths['rail_final']}", "全景输出")}</div>
    """)
    body += section("技术路线", """
      <div class="process">
        <div><strong>视频抽帧</strong><span>从无人机视频中筛选连续帧并整理为拼接序列。</span></div>
        <div><strong>特征匹配</strong><span>SIFT/RootSIFT 提取局部特征，KNN 生成候选匹配。</span></div>
        <div><strong>几何配准</strong><span>RANSAC 估计仿射关系，处理弱连接和视角变化。</span></div>
        <div><strong>全景输出</strong><span>基于全局位姿链生成铁路场景大视野底图。</span></div>
      </div>
    """)
    (PROJECTS_DIR / "railway-stitching.html").write_text(page_shell("铁路无人机视频全景拼接", "铁路巡检视频全景拼接项目", body, active="rail"), encoding="utf-8")


def write_logistics(paths: dict[str, str]) -> None:
    body = immersive_hero(
        "物流大数据垂直大模型",
        "RAG 知识库 · 物流 Skills · 数据工作台",
        "围绕物流公共服务平台中的智能问答、运价分析、路径优化、风险预警和自动周报场景，构建垂直大模型应用原型。",
        f"../{paths['log_bg']}",
        ["RAG", "Streamlit", "路径优化", "风险预警", "自动周报"],
    )
    body += section("数据资产", metric_cards([
        ("模拟运单", "1,766 条"),
        ("仓储记录", "222 条"),
        ("节点风险", "296 条"),
        ("路线网络边", "32 条"),
    ]))
    body += section("总体架构", f"""
      <div class="feature-split">
        <div class="wide-media">{media_tag(f"../{paths['log_arch']}", "项目总体架构")}</div>
        <div class="rich-copy">
          <h3>数据、知识与工具协同</h3>
          <p>项目将结构化物流数据、业务知识库和可计算 Skills 放在同一工作台中，面向问答、预测、路径、风险和报告任务提供统一入口。</p>
          <ul>
            <li>结构化数据用于运价、路线、风险和需求类分析。</li>
            <li>RAG 知识库承接政策、冷链 SOP、铁路通道和数据治理材料。</li>
            <li>确定性工具负责路径计算、指标统计和风险判断，再由应用层组织成可读结果。</li>
          </ul>
        </div>
      </div>
    """)
    body += section("系统展示", f"""
      <div class="gallery">
        {picture_panel(f"../{paths['log_dashboard']}", "运行态势面板")}
        {picture_panel(f"../{paths['log_route']}", "路径优化演示")}
        {picture_panel(f"../{paths['log_rag']}", "RAG 与技能调用链")}
      </div>
    """)
    body += section("应用能力", """
      <div class="text-grid">
        <article class="text-panel"><h3>智能问答</h3><p>面向物流政策、冷链规范、运输通道和业务流程进行知识检索与问答组织。</p></article>
        <article class="text-panel"><h3>运营分析</h3><p>基于运单、仓储和节点数据汇总运价、风险、需求和服务质量指标。</p></article>
        <article class="text-panel"><h3>辅助决策</h3><p>把路径优化、风险预警和自动周报整合为可演示的业务工作台。</p></article>
      </div>
    """)
    (PROJECTS_DIR / "logistics-llm.html").write_text(page_shell("物流大数据垂直大模型", "物流 RAG 与工具调用系统", body, active="logistics"), encoding="utf-8")


def phone_cards(paths: dict[str, str], keys: list[str], labels: list[str]) -> str:
    return "".join(
        f'<div class="phone-card">{media_tag(f"../{paths[key]}", labels[i])}<p>{esc(labels[i])}</p></div>'
        for i, key in enumerate(keys)
    )


def write_miniapps(paths: dict[str, str]) -> None:
    body = immersive_hero(
        "微信小程序展示合集",
        "校园网约车 · 舒腰健脊 · 登山协会",
        "三个小程序覆盖校园出行、康复健康管理和社团活动服务，展示需求拆解、页面流程、状态联动与可运行 Demo 的产品工程能力。",
        f"../{paths['mini_bg']}",
        ["微信小程序", "uni-app / Vue 3", "产品原型", "状态联动"],
    )
    body += section("统一产品链路", """
      <div class="process">
        <div><strong>需求拆解</strong><span>把真实场景拆成角色、任务和页面入口。</span></div>
        <div><strong>数据建模</strong><span>用本地 Mock / storage 承接订单、训练、随访和活动数据。</span></div>
        <div><strong>页面实现</strong><span>按主流程组织首页、详情、表单、记录和个人中心。</span></div>
        <div><strong>交互闭环</strong><span>让创建、加入、打卡、评价、随访和报名动作产生状态联动。</span></div>
      </div>
    """)
    body += section("校园网约车项目", f"""
      <div class="feature-split">
        <div class="picture-panel">
          <div class="picture">{media_tag(f"../{paths['mini_campus_video']}", "校园网约车演示视频", poster=f"../{paths['mini_campus_1']}")}</div>
          <div class="picture-copy"><h3>演示视频</h3><p>覆盖找单、发起拼单、订单详情、行程管理和历史评价。</p></div>
        </div>
        <div class="rich-copy">
          <h3>校园出行拼车闭环</h3>
          <p>以校园到机场/高铁站等高频出行场景为主线，把发起拼单、路线筛选、费用预估、成团进度和个人行程串成完整流程。</p>
          <ul><li>首页承接路线与时段筛选。</li><li>订单页展示座位、费用和成团状态。</li><li>个人中心保留常用路线和历史评价。</li></ul>
        </div>
      </div>
      <div class="phone-grid" style="margin-top:16px">
        {phone_cards(paths, ['mini_campus_1','mini_campus_2','mini_campus_3','mini_campus_4'], ['拼车调度台','发起拼单','订单详情','我的行程'])}
      </div>
    """)
    body += section("舒腰健脊 App / 小程序", f"""
      <div class="feature-split">
        <div class="picture-panel">
          <div class="picture">{media_tag(f"../{paths['mini_shuy_video']}", "舒腰健脊演示视频", poster=f"../{paths['mini_shuy_1']}")}</div>
          <div class="picture-copy"><h3>演示视频</h3><p>患者端与诊疗师端双角色健康管理原型。</p></div>
        </div>
        <div class="rich-copy">
          <h3>康复训练与随访管理</h3>
          <p>围绕腰背健康管理构建患者端训练、评估、随访和医生端工作台，让健康记录、训练方案与随访沟通形成连续链路。</p>
          <ul><li>患者端聚焦训练课程、问诊评估和个人记录。</li><li>医生端聚焦方案、患者管理和随访事项。</li><li>双角色信息结构清晰，适合展示医疗健康类产品原型能力。</li></ul>
        </div>
      </div>
      <div class="phone-grid" style="margin-top:16px">
        {phone_cards(paths, ['mini_shuy_1','mini_shuy_2','mini_shuy_3','mini_shuy_4'], ['患者端首页','训练课程','问诊评估','医生工作台'])}
      </div>
    """)
    body += section("登山协会小程序", f"""
      <div class="feature-split">
        <div class="picture-panel">
          <div class="picture">{media_tag(f"../{paths['mini_mountain_video']}", "登山协会演示视频", poster=f"../{paths['mini_mountain_1']}")}</div>
          <div class="picture-copy"><h3>演示视频</h3><p>围绕社团活动、公告、服务和个人中心组织信息展示。</p></div>
        </div>
        <div class="rich-copy">
          <h3>社团活动与服务展示</h3>
          <p>以活动浏览、协会公告、服务入口和个人页为核心，展示轻量组织类小程序的信息架构、视觉落地和内容组织能力。</p>
          <ul><li>首页突出社团形象和活动入口。</li><li>活动页用于报名、查看行程与了解活动规则。</li><li>服务与公告模块承接社团运营信息。</li></ul>
        </div>
      </div>
      <div class="phone-grid" style="margin-top:16px">
        {phone_cards(paths, ['mini_mountain_1','mini_mountain_2','mini_mountain_3','mini_mountain_4'], ['首页','活动','服务','公告'])}
      </div>
    """)
    (PROJECTS_DIR / "miniapps.html").write_text(page_shell("微信小程序展示合集", "校园网约车、舒腰健脊与登山协会小程序", body, active="miniapps"), encoding="utf-8")


def write_readme() -> None:
    readme = """# 项目作品集

GitHub Pages 静态作品集站点，入口为 `index.html`。

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
- 物流大数据垂直大模型
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
