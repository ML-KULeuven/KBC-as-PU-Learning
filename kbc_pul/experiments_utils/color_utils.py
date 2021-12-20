from typing import Tuple
import matplotlib.colors


def rgb_int_to_float(rgb_ints: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r_fl: float = rgb_ints[0] / 255.0
    g_fl: float = rgb_ints[1] / 255.0
    b_fl: float = rgb_ints[2] / 255.0
    return r_fl, g_fl, b_fl


def rgb_float_to_int(rgb_floats: Tuple[float, float, float]) -> Tuple[int, int, int]:
    r_int: int = int(rgb_floats[0] * 255)
    g_int: int = int(rgb_floats[1] * 255)
    b_int: int = int(rgb_floats[2] * 255)

    return r_int, g_int, b_int


def rgb_ints_to_hex(rgb_ints: Tuple[int, int, int]) -> str:
    return '#%02x%02x%02x' % (rgb_ints[0], rgb_ints[1], rgb_ints[2])


def matplotlib_color_name_to_hex(color_name: str) -> str:
    return matplotlib.colors.to_hex(color_name)


if __name__ == '__main__':
    print(rgb_ints_to_hex((255, 0, 0)))
    print(matplotlib_color_name_to_hex("red"))
