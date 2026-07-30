import os
import cv2, numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from rembg import remove, new_session

CANVAS_W, CANVAS_H = 1094, 1418
PHOTO_W, PHOTO_H = 1094, 1090
PHOTO_X = (CANVAS_W - PHOTO_W) // 2
TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_NAME = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 128)
FONT_ROLE = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 58)

_detector = cv2.FaceDetectorYN_create(TEMPLATE_DIR + "/yunet.onnx", "", (320, 320), score_threshold=0.6)
_rembg_sessions = {}

def cutout_person(pil_img, model="u2net"):
    if model not in _rembg_sessions:
        _rembg_sessions[model] = new_session(model)
    return remove(pil_img, session=_rembg_sessions[model])

def strip_hue(cutout, hue_lo, hue_hi, sat_min=0.25, val_min=0.5, exclude_center=None, exclude_radius=0):
    """Zero out alpha for pixels whose hue falls in [hue_lo, hue_hi] (degrees) -
    used to scrub background objects (e.g. a chair) that survive segmentation.
    If exclude_center/exclude_radius are given, pixels within that radius (e.g. around
    the face) are left untouched even if they match the hue window - skin blush/highlights
    can drift into the same hue range, but the background object sits well outside the face."""
    arr = np.array(cutout).astype(float)
    rgb = arr[:, :, :3] / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc
    safe_delta = np.where(delta == 0, 1, delta)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    h = np.zeros_like(maxc)
    h = np.where(maxc == r, bc - gc, h)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = np.where(delta == 0, 0, (h / 6.0) % 1.0)
    hue = h * 360
    sat = np.where(maxc == 0, 0, delta / np.where(maxc == 0, 1, maxc))
    val = maxc
    mask = (hue >= hue_lo) & (hue <= hue_hi) & (sat > sat_min) & (val > val_min)
    if exclude_center is not None and exclude_radius > 0:
        yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
        dist = np.hypot(xx - exclude_center[0], yy - exclude_center[1])
        mask &= dist > exclude_radius
    out = arr.copy()
    out[mask, 3] = 0
    return Image.fromarray(out.astype("uint8"), "RGBA")

def detect_face(pil_img):
    downscale = max(1, max(pil_img.size) // 900)
    small = pil_img.resize((pil_img.width // downscale, pil_img.height // downscale))
    arr = np.array(small)[:, :, ::-1].copy()
    h, w = arr.shape[:2]
    _detector.setInputSize((w, h))
    retval, faces = _detector.detect(arr)
    if faces is None:
        return None
    face = max(faces, key=lambda f: f[-1])
    return [v * downscale for v in face]

def load_headshot(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img

def crop_to_frame(img, face, target_eye_frac=0.30, target_face_h_frac=0.42):
    fx, fy, fw, fh, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm = face[:14]
    eye_x = (x_re + x_le) / 2
    eye_y = (y_re + y_le) / 2
    mouth_y = (y_rcm + y_lcm) / 2
    aspect = PHOTO_W / PHOTO_H  # target crop aspect ratio (must be preserved, no distortion)

    desired_h = fh / target_face_h_frac

    # Bounds on crop_h so the crop (a) keeps eyes at target_eye_frac from its top,
    # (b) stays within the image vertically, and (c) stays within the image horizontally
    # while keeping eye_x horizontally centered. Take the tightest bound - never distort,
    # never go out of frame; the eye position wins over the exact zoom level when they conflict.
    max_h_top = eye_y / target_eye_frac if target_eye_frac > 0 else float("inf")
    max_h_bottom = (img.height - eye_y) / (1 - target_eye_frac)
    half_w_available = min(eye_x, img.width - eye_x)
    max_h_width = (2 * half_w_available) / aspect

    crop_h = min(desired_h, max_h_top, max_h_bottom, max_h_width)

    # Neck-safety floor: make sure the crop extends well past the mouth so chin/neck/
    # shoulders are visible (a tight face-filling selfie could otherwise crop right at
    # the jaw, leaving a floating "no neck" head). Require the bottom of the crop to sit
    # at least 2.2x the eye-to-mouth distance below the mouth. This can push crop_h above
    # the eye/bounds-derived value above; when it does, neck visibility wins over exact
    # eye_frac placement (the eye anchor below still centers as best it can within bounds).
    eye_to_mouth = max(mouth_y - eye_y, 1)
    neck_margin = 2.2 * eye_to_mouth
    min_neck_h = (mouth_y - eye_y + neck_margin) / (1 - target_eye_frac)
    crop_h = max(crop_h, min_neck_h)
    # Hard cap: crop can never exceed what the source image actually contains
    crop_h = min(crop_h, img.height, img.width / aspect)
    crop_w = crop_h * aspect

    crop_top = eye_y - target_eye_frac * crop_h
    crop_left = eye_x - crop_w / 2
    # Guard against float rounding pushing us a hair out of bounds
    crop_top = max(0, min(crop_top, img.height - crop_h))
    crop_left = max(0, min(crop_left, img.width - crop_w))

    box = (crop_left, crop_top, crop_left + crop_w, crop_top + crop_h)
    cropped = img.crop(tuple(map(round, box)))
    return cropped.resize((PHOTO_W, PHOTO_H), Image.LANCZOS)

def build_card(headshot_path, name, role, out_path, eye_frac=0.34, face_h_frac=0.42,
               model="u2net", cleanup=None, use_cutout=True):
    blank = Image.open(TEMPLATE_DIR + "/blank_template.png").convert("RGBA")
    canvas = blank.copy()

    if headshot_path:
        img = load_headshot(headshot_path)
        face = detect_face(img)
        if face is None:
            raise RuntimeError(f"No face detected in {headshot_path}")
        if use_cutout:
            source = cutout_person(img, model=model)  # RGBA, background removed, same size as img
            if cleanup is not None:
                source = cleanup(source)
        else:
            # Segmentation failed to keep shoulders/torso (e.g. extreme close-up angle) -
            # fall back to a plain opaque crop rather than leaving a floating head.
            source = img.convert("RGBA")
        photo = crop_to_frame(source, face, eye_frac, face_h_frac)
        canvas.paste(photo, (PHOTO_X, 0), photo)
    else:
        generic = Image.open(TEMPLATE_DIR + "/generic_avatar.png").convert("RGBA")
        canvas.paste(generic, (0, 0), generic)

    # overlay logo + nameplate from blank template (crop those regions)
    logo_region = blank.crop((900, 12, 1079, 231))
    canvas.paste(logo_region, (900, 12), logo_region)
    nameplate_region = blank.crop((0, 1090, 1094, 1418))
    canvas.paste(nameplate_region, (0, 1090), nameplate_region)

    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), name, font=FONT_NAME)
    w = bbox[2] - bbox[0]
    draw.text(((CANVAS_W - w) / 2 - bbox[0], 1170), name, font=FONT_NAME, fill=(0, 0, 0, 255))

    bbox2 = draw.textbbox((0, 0), role, font=FONT_ROLE)
    w2 = bbox2[2] - bbox2[0]
    draw.text(((CANVAS_W - w2) / 2 - bbox2[0], 1310), role, font=FONT_ROLE, fill=(0, 0, 0, 255))

    canvas.convert("RGB").save(out_path)
    print("saved", out_path)
