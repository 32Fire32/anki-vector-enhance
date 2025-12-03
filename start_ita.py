# start_ita_stabile.py
import os
# If certifi is available, point gRPC to the certifi CA bundle so
# OpenSSL can verify the server certificate on systems where the
# default root store isn't propagated to the gRPC/openssl build.
try:
    import certifi
    os.environ.setdefault('GRPC_DEFAULT_SSL_ROOTS_FILE_PATH', certifi.where())
except Exception:
    # If certifi isn't available, continue without setting the env var.
    pass

import asyncio
import time
from anki_vector import Robot, exceptions
from vector_ita import RobotItaliano

MAX_RETRIES = 3
TIMEOUT = 60.0

async def main():
    retries = 0
    connected = False
    vector = None

    # Tentativi di connessione
    while retries < MAX_RETRIES and not connected:
        try:
            print(f"[INFO] Tentativo di connessione {retries + 1}/{MAX_RETRIES}...")
            vector = Robot(behavior_activation_timeout=TIMEOUT)
            vector.connect()  # SDK sincrono, non await
            connected = True
            print("[INFO] Connesso a Vector!")
        except exceptions.VectorTimeoutException:
            retries += 1
            print("[WARNING] Timeout durante la connessione. Riprovo tra 5 secondi...")
            time.sleep(5)

    if not connected:
        print("[ERROR] Non è stato possibile connettersi a Vector.")
        return

    # Wrap con il comportamento italiano
    v = RobotItaliano(vector)

    try:
        # Vector parla in italiano
        print("[INFO] Vector dice in italiano...")
        await v.behavior.say_text("Hello Vector, how are you?")

        # Puoi aggiungere qui altri comandi/eventi senza disconnettere
        print("[INFO] Vector rimane connesso per altri comandi...")

        # Mantieni lo script in esecuzione per ascoltare eventi o comandi
        print("[INFO] Premi Ctrl+C per uscire e disconnettere Vector.")
        while True:
            await asyncio.sleep(1)  # loop attivo senza consumare CPU

    except KeyboardInterrupt:
        print("\n[INFO] Ricevuto Ctrl+C. Preparazione alla disconnessione...")

    finally:
        # Disconnessione solo quando chiudi davvero lo script
        print("[INFO] Disconnessione da Vector...")
        vector.disconnect()
        print("[INFO] Chiusura completata.")

if __name__ == "__main__":
    asyncio.run(main())
