from PIL import Image, ImageDraw, ImageFont, ImageOps

CANVAS_W, CANVAS_H = 1000, 1258
DIVIDER_Y = 989
FONT_CAPTION = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 74)

def build_sponsor_card(logo_path, out_path, bg=(255, 255, 255), caption="Match Kit Sponsor",
                        max_w_frac=0.72, max_h_frac=0.62):
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), bg)
    logo = Image.open(logo_path)
    logo = ImageOps.exif_transpose(logo).convert("RGBA")

    max_w = int(CANVAS_W * max_w_frac)
    max_h = int(DIVIDER_Y * max_h_frac)
    scale = min(max_w / logo.width, max_h / logo.height)
    new_size = (max(1, round(logo.width * scale)), max(1, round(logo.height * scale)))
    logo = logo.resize(new_size, Image.LANCZOS)

    x = (CANVAS_W - logo.width) // 2
    y = (DIVIDER_Y - logo.height) // 2
    canvas.paste(logo, (x, y), logo)

    draw = ImageDraw.Draw(canvas)
    line_color = (0, 0, 0) if bg == (255, 255, 255) else (255, 255, 255)
    draw.rectangle([0, DIVIDER_Y, CANVAS_W, DIVIDER_Y + 2], fill=line_color)

    bbox = draw.textbbox((0, 0), caption, font=FONT_CAPTION)
    w = bbox[2] - bbox[0]
    text_color = (0, 0, 0) if bg == (255, 255, 255) else (255, 255, 255)
    draw.text(((CANVAS_W - w) / 2 - bbox[0], 1086), caption, font=FONT_CAPTION, fill=text_color)

    canvas.save(out_path)
    print("saved", out_path)
