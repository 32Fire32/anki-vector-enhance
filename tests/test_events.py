import time
import anki_vector
from anki_vector.events import Events

def on_face_observed(robot, event_type, event):
    print(f"👀 [Face] Visto un volto. Face ID: {event.face_id}")

def on_wake_word(robot, event_type, event):
    print(f"🎤 [Wake Word] Vector ha sentito 'Hey Vector'!")

def on_object_moved(robot, event_type, event):
    print(f"📦 [Oggetto] Un oggetto si sta muovendo.")

def on_object_tapped(robot, event_type, event):
    print(f"👆 [Cubo] Il cubo è stato toccato.")

def on_object_available(robot, event_type, event):
    print(f"🔍 [Cubo] Trovato un cubo disponibile.")

def main():
    with anki_vector.Robot(serial="00701d95", cache_animation_lists=False, enable_face_detection=True, enable_custom_object_detection=True) as robot:
        print("✅ Vector connesso. Sto ascoltando eventi...")
        
        # ATTIVA i vision modes manualmente
        robot.vision.enable_face_detection(True)
        robot.vision.enable_custom_object_detection(True)

        # Mi iscrivo a più eventi contemporaneamente
        # robot.events.subscribe(on_face_observed, Events.robot_observed_face)
        # robot.events.subscribe(on_object_moved, Events.object_moved)
        # robot.events.subscribe(on_object_tapped, Events.object_tapped)
        robot.events.subscribe(on_object_available, Events.object_available)

        # Ascolta gli eventi per 30 secondi
        time.sleep(60)

        print("⏹️ Fine ascolto.")

if __name__ == "__main__":
    main()
