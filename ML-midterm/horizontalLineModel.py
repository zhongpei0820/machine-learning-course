##
#   h(x) = a
#   This class compute the bias, variance and mse of h(x)
#   You can define the number of datasets and number of points per dataset and the result is presented in graph
#   Because datasets are generated randomly, the result may not be 100% precise
##
import matplotlib.pyplot as plt
import numpy as np


class horizontalLineModel:
    def __init__(self, datasets):
        self.datasets = datasets  # training datasets
        self.average_a = sum(
            map(lambda set: float(sum(map(lambda x: self.__functionX(x), set))) / len(set), datasets)) / len(
            datasets)  # average horizontal line over all datasets

    def plotFx(self, accuracy):
        x = np.arange(-1, 1, accuracy)
        plt.plot(x, x ** 2, 'b')

    # plot h(x) = a according to the datasets
    def plotH(self, accuracy):
        averagePerDataset = self.__allAveragePerDataset(self.datasets)
        plt.plot([-1, 1], [averagePerDataset, averagePerDataset],
                 'c')  # plot the line from -1 to 1 with y = a for each dataset
        plt.plot([-1, 1], [self.average_a, self.average_a], 'r', linewidth=2)  # plot the average line for all datasets
        self.plotFx(accuracy)
        plt.show()

    # calculate the bias^2
    def bias(self, accuracy):
        return float(sum(map(lambda x: (self.__functionX(x) - self.average_a) ** 2, np.arange(-1, 1, accuracy)))) / (
        2 / accuracy)

    # caculate the variance
    def var(self):
        return float(sum(map(lambda set: (self.__averagePerDataset(set) - self.average_a) ** 2, self.datasets))) / len(
            self.datasets)

    def mse(self, accuracy):
        return self.bias(accuracy) + self.var()

    # target function defined here
    def __functionX(self, x):
        return x ** 2

    # get all average horizontal lines per dataset
    def __allAveragePerDataset(self, datasets):
        return map(lambda set: self.__averagePerDataset(set), datasets)

    # average horizontal line per dataset
    def __averagePerDataset(self, set):
        return sum(map(lambda x: self.__functionX(x), set)) / len(set)
