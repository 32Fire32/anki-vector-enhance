# import anki_vector
# from anki_vector.util import degrees

# with anki_vector.Robot() as robot:
#     robot.behavior.set_head_angle(degrees(0.0))
#     robot.behavior.set_lift_height(0.0)
#     image = robot.camera.capture_single_image()
#     image.raw_image.show()

import anki_vector

with anki_vector.Robot() as robot:
    print(f"✅ Connesso a {robot.serial}")
    battery = robot.get_battery_state()
    print(f"Livello batteria: {battery.battery_level}")