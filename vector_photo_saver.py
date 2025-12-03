# ...existing code...
import os
import time
import logging
from typing import Optional
from PIL import Image
from anki_vector import Robot
from anki_vector.exceptions import VectorTimeoutException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def capture_image(save_path: Optional[str] = "vector_image.jpg",
                  ip: Optional[str] = None,
                  timeout: int = 10,
                  retries: int = 5,
                  behavior_activation_timeout: int = 300,
                  fallback_to_local: bool = True) -> Image.Image:
    """
    Connetti a Vector, acquisisci un'immagine e la salva (se save_path).
    - timeout: attesa per la fotocamera (secondi)
    - retries: numero di tentativi di connessione
    - behavior_activation_timeout: timeout passato al Robot per l'attivazione dei behavior
    - fallback_to_local: se True, ritorna l'immagine locale esistente in caso di failure
    Solleva RuntimeError se non riesce ad ottenere alcuna immagine e fallback non è possibile.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        logging.info(f"capture_image attempt {attempt}/{retries} (ip={ip})")
        robot = Robot(ip=ip, behavior_activation_timeout=behavior_activation_timeout)
        try:
            robot.connect(timeout=behavior_activation_timeout)
            logging.info("Connesso a Vector, inizializzo camera feed...")
            robot.camera.init_camera_feed()
            camera_image = None
            for _ in range(int(timeout * 2)):
                camera_image = robot.camera.latest_image
                if camera_image is not None:
                    break
                time.sleep(0.5)

            if camera_image is None:
                raise RuntimeError("Timeout camera: nessuna immagine ricevuta")

            image = camera_image.raw_image.convert("RGB")
            out_image = image.copy()
            if save_path:
                out_image.save(save_path)
                logging.info(f"Immagine salvata su {save_path}")

            # chiusura pulita
            try:
                robot.camera.close_camera_feed()
            except Exception:
                logging.debug("Errore chiusura camera_feed", exc_info=True)
            try:
                robot.disconnect()
            except Exception:
                logging.debug("Errore disconnect", exc_info=True)

            return out_image

        except VectorTimeoutException as e:
            last_exc = e
            logging.warning(f"VectorTimeoutException {attempt}/{retries}: {e}")
            try:
                robot.disconnect()
            except Exception:
                pass
            # backoff semplice
            time.sleep(2 * attempt)
            continue
        except Exception as e:
            last_exc = e
            logging.error(f"Errore durante capture_image: {e}", exc_info=True)
            try:
                robot.camera.close_camera_feed()
            except Exception:
                pass
            try:
                robot.disconnect()
            except Exception:
                pass
            break

    # Tutti i retry falliti
    if fallback_to_local and save_path and os.path.exists(save_path):
        logging.warning(f"Ritorno immagine locale di fallback: {save_path}")
        return Image.open(save_path).convert("RGB")

    raise last_exc or RuntimeError("Connessione a Vector fallita dopo i retry")
# ...existing code...