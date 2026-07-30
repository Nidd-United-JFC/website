# Coach & sponsor card generator

Generates placeholder coach headshot cards and sponsor cards in the club's
striped template style (same look as e.g. `images/nigel-corner-coach.png` or
`images/C4U-Match-kit-sponsor-2026-27.png`), from a raw headshot/logo photo.

## Setup (one-off)

```
cd tools/coach-cards
python3 -m venv venv
source venv/bin/activate
pip install pillow rembg onnxruntime opencv-python-headless
```

Uses macOS's built-in Helvetica Neue font (`/System/Library/Fonts/HelveticaNeue.ttc`),
so this is Mac-only as written.

## Adding a new coach headshot

```python
import build_coach as bc

# With a real headshot photo - cuts the person out of their background and
# composites them over the club stripes, framed with eyes in the top third.
bc.build_card(
    "/path/to/headshot.jpg",
    "Coach Name", "Coach",
    "../../images/coach-name-coach.png",
)

# No headshot on file - reuses the generic silhouette placeholder graphic
# (same one already used across the site, e.g. images/andrew-brown-coach.png).
bc.build_card(None, "Coach Name", "Coach", "../../images/coach-name-coach.png")
```

Filename convention: `firstname-lastname-coach.png`, lowercase-hyphenated,
dropped straight into `images/`.

### If the auto-cutout goes wrong

- **Background object bleeds through (e.g. a chair behind the head):** try
  `model="isnet-general-use"` or `model="u2net_human_seg"` instead of the
  default `"u2net"`. If a specific colour still survives, use `strip_hue()`
  with `exclude_center`/`exclude_radius` set to the face position so you
  don't accidentally erase skin tones too (see git history on this file for
  a worked example - the Ian Dunwell pink-chair fix).
- **Shoulders/torso get cut away entirely (floating head), usually on very
  tight/extreme-angle selfies:** the segmentation model failed outright.
  Try the other two models first (`u2net_human_seg` often does better on
  awkward angles than the default). If none of them keep the torso, fall
  back to `use_cutout=False` - this pastes a plain rectangular crop with the
  photo's own background instead of a cutout (see the Andy Prest fix).
- **Eyes too high/low:** tweak `eye_frac` (default 0.34, fraction of the
  frame height down from the top) and `face_h_frac` (default 0.42, how much
  of the frame height the face fills) on the `build_card()` call.

## Adding a new sponsor card

```python
from build_sponsor_card import build_sponsor_card

build_sponsor_card(
    "/path/to/sponsor-logo.png",
    "../../images/sponsor-name-Match-kit-sponsor-2026-27.png",
    caption="Match Kit Sponsor",  # or "Training Support Sponsor" etc.
)
```

Matches the existing card format used across the site (e.g.
`$DUBCAT-Match-kit-sponsor-2025-26.png`, `JRGraham-Match-kit-sponsor-2025-26.png`):
logo centred on a white (or black, pass `bg=(0,0,0)`) background, a divider
line, and the caption underneath.

Then wire the new image into `teams.html` (and `our-sponsors.html` /
`index.html` if it's a new sponsor, not just a returning one) the same way
the existing sponsor `<a>` tags are set up.

## Files in this folder

- `build_coach.py` - coach card generator (face detection, background
  cutout, template compositing)
- `build_sponsor_card.py` - sponsor card generator
- `blank_template.png` - the striped background + crest + blank nameplate,
  extracted from `Website/Images/NU-Headshot-Club Background-Template.psd`
- `generic_avatar.png` - the generic "no headshot on file" silhouette,
  extracted from the site's own `images/andrew-brown-coach.png`
- `yunet.onnx` - face detection model (OpenCV FaceDetectorYN / YuNet)
