# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    data = np.reshape(np.loadtxt("Data/M2.csv",
                                 delimiter=",", dtype=float), (-1, 5))
    out = data[:, 0]

    a = 3

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
