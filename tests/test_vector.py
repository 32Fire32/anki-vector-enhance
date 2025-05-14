import anki_vector
from anki_vector.util import degrees

def main():
    # Usa il blocco with per una gestione automatica della connessione
    with anki_vector.AsyncRobot(serial="00701d95", cache_animation_lists=False) as robot:
        # Avvia le azioni asincrone
        say_future = robot.behavior.say_text("Ciao, come stai?")
        # Assicurati che tutte le azioni siano completate prima di continuare
        say_future.result()  # Aspetta che Vector abbia detto il testo
        
        turn_future = robot.behavior.turn_in_place(degrees(360))
        turn_future.result()  # Aspetta che Vector abbia completato il giro
        
        # Al termine delle azioni, il robot verrà disconnesso automaticamente al termine del blocco with
        # Non è necessario chiamare robot.disconnect() esplicitamente

if __name__ == "__main__":
    main()