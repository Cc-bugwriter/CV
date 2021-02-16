import numpy as np
import matplotlib.pyplot as plt

#
# Problem 1
#
import problem1 as p1

def problem1():

    def evaluation(x, y, nn):
        m = x.shape[1]
        pred = np.zeros((1, m))
        output = nn.forward(x)

        for i in range(0, output.shape[1]):
            if output[0, i] > 0.5:
                pred[0, i] = 1
            else:
                pred[0, i] = 0

        print("Accuracy: " + str(np.sum((pred == y) / float(m))))
        return np.array(pred[0], dtype=np.int), (pred == y)[0], np.sum(
            (pred == y) / float(m)) * 100

    # fix some random seed
    np.random.seed(187)

    # load data
    data= np.load('data/p1.npz')
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    print("Number of training images: %d" % X_train.shape[1])
    print("Number of test images: %d" % X_test.shape[1])

    # set network and training parameters
    net_structure = [4096, 192, 1]
    lr = 1e-4
    batch_size = 32
    num_epochs = 20

    # initialize network
    nn = p1.Network(net_structure)
    # train network
    nn.train(X_train, Y_train, lr, batch_size, num_epochs)

    # evaluate performance on the training set
    plt.figure(figsize=(10, 20))
    plt.subplots_adjust(wspace=0, hspace=0.15)
    pred_train, correctly_classified, accuracy = evaluation(X_train, Y_train, nn)
    for i in range(X_train.shape[1]):
        ax = plt.subplot(21, 10, i + 1)

        x_data = X_train[:,i].reshape(64, 64)

        if not correctly_classified[i]:
            im = ax.imshow(x_data, cmap='hot')
        else:
            im = ax.imshow(x_data, cmap='Greys_r')

        plt.xticks([])
        plt.yticks([])
        plt.suptitle(
            "Training set, number of images: %d\n Accuracy: %.2f%%, misclassified examples are represented in a red-yellow colormap."
            % (X_train.shape[1], accuracy),
            fontsize=12)


    # evaluate performance on the validation set
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(wspace=0, hspace=0.1)
    predicted, correctly_classified, accuracy = evaluation(X_test, Y_test, nn)
    for i in range(X_test.shape[1]):
        ax = plt.subplot(8, 8, i + 1)

        x_data = X_test[:,i].reshape(64, 64)

        if not correctly_classified[i]:
            im = ax.imshow(x_data, cmap='hot')
        else:
            im = ax.imshow(x_data, cmap='Greys_r')

        plt.xticks([])
        plt.yticks([])
        plt.suptitle(
            "Test set, number of images: %d\n Accuracy: %.2f%%, misclassified examples are represented in a red-yellow colormap."
            % (X_test.shape[1], accuracy),
            fontsize=12)

    plt.show()

if __name__ == "__main__":
    problem1()

    # def evaluation(x, y, nn):
    #     m = x.shape[1]
    #     pred = np.zeros((1, m))
    #     output = nn.forward(x)
    #
    #     for i in range(0, output.shape[1]):
    #         if output[0, i] > 0.5:
    #             pred[0, i] = 1
    #         else:
    #             pred[0, i] = 0
    #
    #     print("Accuracy: " + str(np.sum((pred == y) / float(m))))
    #     return np.array(pred[0], dtype=np.int), (pred == y)[0], np.sum(
    #         (pred == y) / float(m)) * 100
    #
    # # fix some random seed
    # np.random.seed(187)
    #
    # # load data
    # data= np.load('data/p1.npz')
    # X_train = data['X_train']
    # Y_train = data['Y_train']
    # X_test = data['X_test']
    # Y_test = data['Y_test']
    #
    # print("Number of training images: %d" % X_train.shape[1])
    # print("Number of test images: %d" % X_test.shape[1])
    #
    # # set network and training parameters
    # net_structure = [4096, 192, 1]
    # lr = 1e-4
    # batch_size = 32
    # num_epochs = 20
    #
    # # initialize network
    # nn = p1.Network(net_structure)
    # # train network
    # nn.train(X_train, Y_train, lr, batch_size, num_epochs)