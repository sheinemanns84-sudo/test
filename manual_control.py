import time
import numpy as np
import pybullet as p
from backhoe_env import BackhoeHydraulicEnv


def main():
    env = BackhoeHydraulicEnv(render=True)
    obs, _ = env.reset()

    print("\nSteuerung mit Pfeiltasten:")
    print("  \u2191 / \u2193   = Boom hoch / runter")
    print("  \u2190 / \u2192   = Stick raus / rein")
    print("  Pos1 / Ende = Turret links / rechts")
    print("  Bild\u2191 / Bild\u2193 = Bucket auf / zu")
    print("STRG+C zum Abbrechen.\n")

    try:
        while True:
            keys = p.getKeyboardEvents()
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            step = 0.8  # Schrittweite (zwischen -1.0 und 1.0)

            if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                action[1] = +step
            elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                action[1] = -step

            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                action[2] = +step
            elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                action[2] = -step

            if p.B3G_HOME in keys and keys[p.B3G_HOME] & p.KEY_IS_DOWN:
                action[0] = +step
            elif p.B3G_END in keys and keys[p.B3G_END] & p.KEY_IS_DOWN:
                action[0] = -step

            if p.B3G_PAGE_UP in keys and keys[p.B3G_PAGE_UP] & p.KEY_IS_DOWN:
                action[3] = +step
            elif p.B3G_PAGE_DOWN in keys and keys[p.B3G_PAGE_DOWN] & p.KEY_IS_DOWN:
                action[3] = -step

            env.step(action)
            time.sleep(env.time_step)
    except KeyboardInterrupt:
        print("Manuelle Steuerung beendet.")

    env.close()


if __name__ == "__main__":
    main()
