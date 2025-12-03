import argparse
from pathlib import Path
import logging
import anki_vector

from vector_photo_saver import capture_image
from vector_photo_analyzer import analyze_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser(description="Capture a photo from Vector and analyze it with YOLOv5")
    p.add_argument('--ip', default=None, help='IP address of Vector (optional)')
    p.add_argument('--out', default='vector_image_filtered.jpg', help='output annotated image path')
    p.add_argument('--img', default='vector_image.jpg', help='temporary image save path')
    p.add_argument('--model', default='yolov5s', help='yolov5 model name (yolov5n/yolov5s/etc)')
    p.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    p.add_argument('--size', type=int, default=640, help='inference image size')
    p.add_argument('--timeout', type=int, default=15, help='camera timeout (seconds)')
    p.add_argument('--retries', type=int, default=3, help='connect retries')
    p.add_argument('--behavior-timeout', type=int, default=120, help='behavior activation timeout (seconds)')
    p.add_argument('--no-save-fallback', dest='fallback', action='store_false', help='do not fallback to existing local image on capture failure')
    args = p.parse_args()

    # capture
    logging.info('Capturing image from Vector...')
    try:
        # Initialize robot connection
        with anki_vector.Robot(args.ip) as robot:
            # Play a photo-taking animation trigger
            logging.info('Playing photo animation...')
            robot.anim.play_animation_trigger('OnboardingReactToFaceHappy', ignore_body_track=True)
            
            # Capture the image
            img = capture_image(save_path=args.img,
                              ip=args.ip,
                              timeout=args.timeout,
                              retries=args.retries,
                              fallback_to_local=args.fallback)
            logging.info(f'Image captured and saved to {args.img}')
    except Exception as e:
        logging.error(f'Capture failed: {e}')
        return

    # analyze
    logging.info('Analyzing image...')
    try:
        df, annotated = analyze_image(img,
                                      out_path=args.out,
                                      model_name=args.model,
                                      conf_threshold=args.conf,
                                      size=args.size,
                                      save=True)
        logging.info(f'Analysis complete. Annotated image saved to {args.out}')
        print(df)
    except Exception as e:
        logging.error(f'Analysis failed: {e}')


if __name__ == '__main__':
    main()
