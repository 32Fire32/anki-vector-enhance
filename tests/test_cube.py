import anki_vector
import traceback
from anki_vector.connection import VectorControlTimeoutException

def connect_cube():
    try:
        with anki_vector.Robot(cache_animation_lists=False) as robot:
            print("Verifico stato del cubo...")

            if robot.world.connected_light_cube:
                print("✅ Cubo già connesso.")
            else:
                print("🟡 Cubo NON connesso. Provo a connetterlo...")
                robot.world.forget_preferred_cube()
                robot.world.connect_cube()  # Non usare 'success', non serve
                print("✅ Comando di connessione al cubo inviato.")
    except VectorControlTimeoutException as e:
        print("⚠️ Timeout nella connessione con Vector:", e)
    except Exception as e:
        print("⚠️ Errore imprevisto:")
        traceback.print_exc()

if __name__ == "__main__":
    connect_cube()