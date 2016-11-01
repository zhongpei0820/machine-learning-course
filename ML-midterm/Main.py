import horizontalLineModel
import linearModel
import random

numberOfDatasets = 500
numberOfPointPerDataset = 2
accuracy = 0.01

def main() :

    datasets = [[random.uniform(-1,1) for i in range(0,numberOfPointPerDataset)] for j in range(0,numberOfDatasets)]

    model1 = horizontalLineModel.horizontalLineModel(datasets)
    print "The bias of model1 is approximately {0:.3f}".format(model1.bias(accuracy))  # get the bias of h1(x)
    print "The variance of model1 is approximately {0:.3f}".format(model1.var())  # gei the variance of h1(x)
    print "The MSE is approximately {0:.3f}".format(model1.mse(accuracy))  # print the mse
    model1.plotH(accuracy)

    model2 = linearModel.linearModel(datasets)
    print "The bias of model2 is approximately {0:.3f}".format(model2.bias(accuracy))  # get the bias of h1(x)
    print "The variance of model2 is approximately {0:.3f}".format(model2.var(accuracy))  # gei the variance of h1(x)
    print "The MSE is approximately {0:.3f}".format(model2.mse(accuracy))  # print the mse
    model2.plotH(accuracy)


if __name__ == '__main__':
    main()