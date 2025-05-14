import anki_vector

def main():
    with anki_vector.Robot(serial="00701d95", cache_animation_lists=False) as robot:
        battery_state = robot.get_battery_state()

        print(f"Voltaggio batteria: {battery_state.battery_volts:.2f}V")
        print(f"Livello batteria: {battery_state.battery_level}")
        print(f"Sta caricando: {battery_state.is_charging}")
        print(f"Sulla base di carica: {battery_state.is_on_charger_platform}")

        if battery_state.battery_level == 1:
            print("⚡ Batteria BASSA! Meglio mettere in carica.")
        elif battery_state.battery_level == 3:
            print("🔋 Batteria PIENA!")
        else:
            print("✅ Batteria OK, puoi continuare!")

if __name__ == "__main__":
    main()