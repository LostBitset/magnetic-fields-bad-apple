from simframe import SimFrame
import cv2

# Frames range from 1 to 6571

def process_frame(n):
    pfx = f"[process_frame] (Frame {n}): "
    print(pfx + "Loading frame...")
    sim = SimFrame(f"frames/BadApple_{n}.jpg")
    print(pfx + "Calculating B-field...")
    sim.bake_b_field()
    print(pfx + "Drawing B-field...")
    image = sim.draw_b_field()
    print(pfx + "Saving new image...")
    status = cv2.imwrite(f"outframes/BadAppleBField_{n}.jpg", image)
    if not status:
        raise Exception("ERROR!! Failed to save image.")
    print(pfx + "All done.")

process_frame(1)
process_frame(810)

