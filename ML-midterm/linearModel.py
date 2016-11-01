##
#   h(x) = ax + b
#   This program compute the bias, variance and mse of h(x)
#   You can define the number of datasets and number of points per dataset and the result is presented in graph
#   Because datasets are generated randomly, the result may not be 100% precise
##
import matplotlib.pyplot as plt
import numpy as np


class linearModel:
    def __init__(self, datasets):
        self.datasets = datasets  # training datasets
        average_ab = map(lambda x: sum(x) / len(x), zip(*map(lambda set: self.__averagePerDataset(set), self.datasets)))
        self.average_a = average_ab[0]
        self.average_b = average_ab[1]

    def plotFx(self, accuracy):
        x = np.arange(-1, 1, accuracy)
        plt.plot(x, x ** 2, 'b')

    def plotXY(self):
        plt.plot([-1, 1], [0, 0], 'k')
        plt.plot([0, 0], [-1.5, 1.5], 'k')
        plt.axis([-1, 1, -1.5, 1.5])

    def plotH(self, accuracy):
        for averagePerDataset in self.__allAveragePerDataset():
            a, b = averagePerDataset[0], averagePerDataset[1]
            plt.plot([-1, 1], [-a + b, a + b], 'c')
        plt.plot([-1, 1], [-self.average_a + self.average_b, self.average_a + self.average_b], 'r', linewidth=2)
        self.plotFx(accuracy)
        self.plotXY()
        plt.show()

    # calculate the bias^2
    def bias(self, accuracy):
        return float(sum(map(lambda x: (self.__functionX(x) - self.average_a * x + self.average_b) ** 2,
                             np.arange(-1, 1, accuracy)))) / (
                   2 / accuracy)

    # caculate the variance
    def var(self, accuracy):
        var = 0
        for set in self.datasets:
            average_ab = self.__averagePerDataset(set)
            a, b = average_ab[0], average_ab[1]
            for x in np.arange(-1, 1, accuracy):
                var += ((a - self.average_a) * x + (b - self.average_b)) ** 2
        var /= len(self.datasets) * (2 / accuracy)
        return var

    def mse(self, accuracy):
        return self.bias(accuracy) + self.var(accuracy)

    def __functionX(self, x):
        return x ** 2

    def __allAveragePerDataset(self):
        return map(lambda set: self.__averagePerDataset(set), self.datasets)

    # average horizontal line per dataset
    def __averagePerDataset(self, set):
        avg_x = sum(set) / len(set)
        avg_y = sum(map(lambda x: self.__functionX(x), set)) / len(set)
        avg_xy = sum(map(lambda x: x * self.__functionX(x), set)) / len(set)
        avg_xsq = sum(map(lambda x: x ** 2, set)) / len(set)
        a = (avg_xy - avg_x * avg_y) / (avg_xsq - avg_x ** 2)
        b = avg_y - avg_x * a
        return [a, b]
