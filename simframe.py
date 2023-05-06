from PIL import Image
import numpy as np

DEFAULT_CURRENT = 1  # Amps

class SimFrame:
    def __init__(self, path, current=DEFAULT_CURRENT):
        pic = Image.open(path, "r")
        self.arr = np.asarray(pic)[:,:,0] > 200
        self.nocurrent = self.arr.sum() == 0
        if self.nocurrent:
            self.current_density = None
            return
        cross_sectional_area = self.arr.sum() / self.arr.size
        self.current_density = current / cross_sectional_area

    def __str__(self):
        cd = f"J = {self.current_density}"
        return f"< SimFrame {self.arr.shape} with {cd} >"

    def b_field(self):
        field = np.zeros((*self.arr.shape, 2))
        # TODO the actual thing lol
        return field

test = SimFrame("frames/BadApple_358.jpg")

if __name__ == "__main__":
    print(test)

