from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .gemini_analyzer import GENERATION_ROUTE_LABELS, SCENE_TYPE_LABELS
from .models import Segment

TYPE_LABELS = {
    "with_human": "有人",
    "without_human": "无人",
}
CONTACT_SHEET_TYPE_LABELS = {
    "with_human": "human",
    "without_human": "no_human",
}


def make_contact_sheet(segments: list[Segment], output_path: str | Path, thumb_width: int = 180) -> None:
    frames = [segment for segment in segments if segment.representative_frame and Path(segment.representative_frame).exists()]
    if not frames:
        return

    thumb_height = int(thumb_width * 16 / 9)
    label_height = 58
    cols = 4
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_width, rows * (thumb_height + label_height)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for index, segment in enumerate(frames):
        image = Image.open(segment.representative_frame).convert("RGB")
        image.thumbnail((thumb_width, thumb_height))
        cell = Image.new("RGB", (thumb_width, thumb_height), (0, 0, 0))
        cell.paste(image, ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2))

        x = (index % cols) * thumb_width
        y = (index // cols) * (thumb_height + label_height)
        sheet.paste(cell, (x, y))
        label = f"{segment.segment_id} {segment.start:.1f}-{segment.end:.1f}s"
        draw.text((x + 6, y + thumb_height + 6), label, fill=(255, 255, 255))
        draw.text((x + 6, y + thumb_height + 25), CONTACT_SHEET_TYPE_LABELS.get(segment.segment_type, segment.segment_type), fill=(145, 190, 255))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_html_report(
    segments: list[Segment],
    output_path: str | Path,
    gemini_analysis: list[dict] | None = None,
) -> None:
    """生成 HTML 分析报告，可选含 Gemini 理解结果。"""
    # 建立 gemini 索引（按 segment_id）
    gemini_map: dict[str, dict] = {}
    if gemini_analysis:
        for item in gemini_analysis:
            sid = item.get("segment_id", "")
            if sid:
                gemini_map[sid] = item

    has_gemini = bool(gemini_map)

    rows = []
    for segment in segments:
        frame_html = ""
        if segment.representative_frame:
            rel = Path(segment.representative_frame).name
            frame_html = f'<img src="frames_1fps/{rel}" />'

        gemini = gemini_map.get(segment.segment_id)
        gemini_cells = _gemini_html_cells(gemini) if gemini else None

        if gemini_cells and has_gemini:
            # 有 Gemini 结果 → 扩展列
            rows.append(
                "<tr>"
                f"<td>{frame_html}</td>"
                f"<td>{segment.segment_id}</td>"
                f"<td>{segment.start:.2f}</td>"
                f"<td>{segment.end:.2f}</td>"
                f"<td>{segment.duration:.2f}</td>"
                f"<td>{TYPE_LABELS.get(segment.segment_type, segment.segment_type)}</td>"
                f"<td>{segment.confidence:.2f}</td>"
                f"<td>{'<br>'.join(segment.detected)}</td>"
                f"<td>{gemini_cells['scene_type_label']}</td>"
                f"<td>{gemini_cells['route_label']}</td>"
                f"<td>{gemini_cells['description']}</td>"
                f"<td>{gemini_cells['needs_review']}</td>"
                f"<td>{segment.recommended_tech}</td>"
                f"<td>{'<br>'.join(segment.notes)}</td>"
                "</tr>"
            )
        else:
            # 无 Gemini → 基础列
            rows.append(
                "<tr>"
                f"<td>{frame_html}</td>"
                f"<td>{segment.segment_id}</td>"
                f"<td>{segment.start:.2f}</td>"
                f"<td>{segment.end:.2f}</td>"
                f"<td>{segment.duration:.2f}</td>"
                f"<td>{TYPE_LABELS.get(segment.segment_type, segment.segment_type)}</td>"
                f"<td>{segment.confidence:.2f}</td>"
                f"<td>{'<br>'.join(segment.detected)}</td>"
                f"<td>{segment.recommended_tech}</td>"
                f"<td>{'<br>'.join(segment.notes)}</td>"
                "</tr>"
            )

    # 表头定义
    base_headers = ["Frame", "ID", "Start", "End", "Dur", "类别", "Conf", "Detected"]
    gemini_headers = ["场景类型", "生成路线", "Gemini描述", "需复审"]
    right_headers = ["Recommended Tech", "Notes"]

    if has_gemini:
        all_headers = base_headers + gemini_headers + right_headers
    else:
        all_headers = base_headers + right_headers

    header_html = "".join(f"<th>{h}</th>" for h in all_headers)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Video Segmentation Report</title>
  <style>
    body {{ font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; background: #111; color: #eee; margin: 24px; }}
    h1 {{ font-size: 22px; margin-bottom: 6px; }}
    .subtitle {{ color: #999; font-size: 13px; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #333; padding: 8px; vertical-align: top; }}
    th {{ background: #1d2b4f; position: sticky; top: 0; white-space: nowrap; }}
    img {{ width: 120px; display: block; }}
    td:nth-child(6) {{ color: #8fc1ff; font-weight: 700; }}
    .gemini-tag {{
      display: inline-block; padding: 2px 8px; border-radius: 3px;
      font-size: 12px; font-weight: 700; margin-right: 4px;
    }}
    .tag-human-driver {{ background: #1a3a5c; color: #7fc1ff; }}
    .tag-image-to-video {{ background: #2d4a1a; color: #8fdf5f; }}
    .tag-review {{ background: #5c1a1a; color: #ff7f7f; }}
    .tag-product {{ background: #3a2d1a; color: #ffc107; }}
    .tag-graphic {{ background: #2d1a3a; color: #c77fff; }}
    .tag-default {{ background: #333; color: #ccc; }}
    .desc {{ color: #ccc; font-size: 12px; max-width: 250px; }}
    .scene-type {{ font-weight: 600; white-space: nowrap; }}
    .review-badge {{
      display: inline-block; padding: 1px 8px; border-radius: 10px;
      font-size: 11px; font-weight: 700;
    }}
    .badge-yes {{ background: #5c1a1a; color: #ff7f7f; }}
    .badge-no {{ background: #1a3a1a; color: #7fdf7f; }}
  </style>
</head>
<body>
  <h1>Video Segmentation Report</h1>
  <div class="subtitle">"""
    if has_gemini:
        html += "含 Gemini 视觉理解 | "
    html += f"""{len(segments)} 个片段</div>
  <table>
    <thead>
      <tr>{header_html}</tr>
    </thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    Path(output_path).write_text(html, encoding="utf-8")


def _gemini_html_cells(gemini: dict) -> dict[str, str]:
    """从 Gemini 结果生成 HTML 列内容。"""
    st = gemini.get("scene_type", "unknown")
    route = gemini.get("generation_route", "manual_review")
    desc = gemini.get("description", "")
    needs_review = gemini.get("needs_manual_review", True)

    # 场景类型标签
    st_label = SCENE_TYPE_LABELS.get(st, "未知")
    scene_type_html = f'<span class="scene-type">{st_label}</span>'

    # 生成路线标签（带颜色）
    route_label = GENERATION_ROUTE_LABELS.get(route, "人工确认")
    tag_class = {
        "human_driver": "tag-human-driver",
        "human_driver_with_product_reference": "tag-human-driver",
        "image_to_video": "tag-image-to-video",
        "product_replication": "tag-product",
        "graphic_animation": "tag-graphic",
        "manual_review": "tag-review",
    }.get(route, "tag-default")
    route_html = f'<span class="gemini-tag {tag_class}">{route_label}</span>'

    # 需要复审 badge
    if needs_review:
        review_html = '<span class="review-badge badge-yes">⚠ 需要复审</span>'
    else:
        review_html = '<span class="review-badge badge-no">✓ 自动确认</span>'

    return {
        "scene_type_label": scene_type_html,
        "route_label": route_html,
        "description": f'<div class="desc">{desc}</div>',
        "needs_review": review_html,
    }
