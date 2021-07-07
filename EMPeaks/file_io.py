# License: BSD-3-clause
# Copyright © 2020 National Institute of Advanced Industrial Science and Technology (AIST)

import numpy as np
import matplotlib.pyplot as plt


class FileIO:
    def __init__(self, file, file_type):
        self.f_name = file
        self.f_type = file_type
        self.data2d = self.read_file()

    def read_file(self):
        if self.f_type == 'GFET':
            return self.read_GFET()
        else:
            print('ERROR. There are no file type settings. This type cannot read by this method.')
            quit()

    def read_GFET(self):
        f = open(self.f_name, "r")
        L = f.readlines()
        f.close()

        L2 = list(map(lambda s: s.strip(), L))
        i_init = L2.index('[Data 1]')
        dat_s = L2[i_init + 1:-1]
        dat_s = list(map(lambda s: s.split(), dat_s))

        return np.array(dat_s, dtype='float64').T

    def plot(self):
        fig = plt.figure()
        plt.scatter(self.data2d[0], self.data2d[1])


def main():
    io = FileIO('GFET0126_25V_C1s.txt', 'GFET')
    print(io.data2d)


if __name__ == '__main__':
    main()
