import logging
from typing import Optional, Tuple, Any

import torch
from PIL import Image, ImageDraw, ImageFont

logging.getLogger('anki_vector').setLevel(logging.WARNING)
logging.getLogger('connection').setLevel(logging.WARNING)
logging.getLogger('aiogrpc').setLevel(logging.WARNING)


def _get_font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None


def analyze_image(image,
                  out_path: Optional[str] = "vector_image_filtered.jpg",
                  model_name: str = 'yolov5s',
                  conf_threshold: float = 0.5,
                  size: int = 640,
                  device: Optional[str] = None,
                  save: bool = True) -> Tuple[Any, Any]:
    """
    Analyze a PIL Image (or image path) with a YOLOv5 model.

    Returns (dataframe, image_with_boxes).
    - image: PIL Image or path to image file
    - out_path: where to save the image with boxes (if save=True)
    - model_name: 'yolov5s','yolov5n', etc.
    - conf_threshold: confidence threshold
    - size: inference size
    - device: device string e.g. 'cpu' or 'cuda'
    """
    # load image if needed
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image

    # load model (uses torch.hub)
    model = torch.hub.load('ultralytics/yolov5', model_name, trust_repo=True)
    if device:
        try:
            model.to(device)
        except Exception:
            logging.debug("Unable to move model to device", exc_info=True)
    # set threshold
    model.conf = conf_threshold

    # run inference
    with torch.no_grad():
        results = model(img, size=size)

    df = results.pandas().xyxy[0]

    # draw boxes
    draw = ImageDraw.Draw(img)
    font = _get_font(size=14)

    for _, row in df.iterrows():
        x1 = int(round(row['xmin'])); y1 = int(round(row['ymin']))
        x2 = int(round(row['xmax'])); y2 = int(round(row['ymax']))
        label = f"{row.get('name', '')} {row.get('confidence', 0):.2f}"

        # measure text
        if font is not None:
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
            except Exception:
                try:
                    tw, th = font.getsize(label)
                except Exception:
                    tw, th = len(label) * 6, 10
        else:
            tw, th = len(label) * 6, 10

        text_x = x1
        text_y = y1 - th - 2 if (y1 - th - 2) >= 0 else y1 + 2

        # draw box then label background and text
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.rectangle([text_x - 1, text_y - 1, text_x + tw + 1, text_y + th + 1], fill="black")
        draw.text((text_x, text_y), label, fill="white", font=font)

    if save and out_path:
        img.save(out_path)
        logging.info(f"Saved annotated image to {out_path}")

    return df, img


if __name__ == "__main__":
    # simple demo when run directly: capture then analyze
    from vector_photo_saver import capture_image

    img = capture_image()
    df, annotated = analyze_image(img)
    print(df)