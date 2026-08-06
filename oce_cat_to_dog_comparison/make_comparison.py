from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "oce_cat_to_dog_comparison"
PROMPT = "A small furry household pet with pointed ears, whiskers, and a flexible tail."


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


original = Image.open(RESULT_DIR / "W_0.png").convert("RGB")
edited = Image.open(RESULT_DIR / "W.png").convert("RGB")

canvas = Image.new("RGB", (1024, 684), "white")
draw = ImageDraw.Draw(canvas)
draw.text((20, 16), "OCE cat -> dog", fill="black", font=font(28))
draw.text((20, 53), f'Prompt: "{PROMPT}"', fill="#333333", font=font(16))

canvas.paste(original, (0, 130))
canvas.paste(edited, (512, 130))

draw.rectangle((0, 88, 511, 130), fill="#e8eef7")
draw.rectangle((512, 88, 1023, 130), fill="#e9f6eb")
draw.text((18, 96), "W_0 (original)", fill="black", font=font(22))
draw.text((530, 96), "W (OCE edited)", fill="black", font=font(22))
draw.rectangle((0, 642, 1023, 683), fill="white")
draw.text((18, 652), "CLIP cat 70.71% | dog 29.29%", fill="#333333", font=font(16))
draw.text((530, 652), "CLIP cat 23.21% | dog 76.79%", fill="#333333", font=font(16))

canvas.save(ROOT / "OCE_cat_to_dog_W0_vs_W.png")
