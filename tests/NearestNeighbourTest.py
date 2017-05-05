import cs231n.classifiers.k_nearest_neighbor as knn
import cs231n.data_utils as datautils
import unittest
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class NearestNeighbour():
    def train(self, Xtr_rows, Ytr):
        self.Xtr_rows = Xtr_rows
        self.Ytr = Ytr

    def predict(self, Xte_rows):
        #the train data is an array of size numTrainingExample x trainingArraySize. in our case: 50,000 x 3072
        #each row of Xte_rows is an array of size 3072
        #compute the sumb of abs difference between test row and each row of train array as an 50,000 elements array
        num_tests = Xte_rows.shape[0]
        res = np.zeros(num_tests, dtype = self.Ytr.dtype) # the result is an array with the best classification for each example
        for test_index in range(num_tests):
            test_row = Xte_rows[test_index,:]
            distances = np.sum(abs(self.Xtr_rows - test_row), axis=1)
            best_index = np.argmin(distances)
            best_classification = self.Ytr[best_index]
            res[test_index] = best_classification
            print('Classified test image ', test_index, 'as category ', best_classification)
        return res

class KNearestNeighbour():
    def train(self, Xtr_rows, Ytr):
        self.Xtr_rows = Xtr_rows
        self.Ytr = Ytr

    def predict(self, Xte_rows,k):
        #the train data is an array of size numTrainingExample x trainingArraySize. in our case: 50,000 x 3072
        #each row of Xte_rows is an array of size 3072
        #compute the sumb of abs difference between test row and each row of train array as an 50,000 elements array
        num_tests = Xte_rows.shape[0]
        res = np.zeros(num_tests, dtype = self.Ytr.dtype) # the result is an array with the best classification for each example
        for test_index in range(num_tests):
            test_row = Xte_rows[test_index,:]
            distances = np.sum(abs(self.Xtr_rows - test_row), axis=1)
            best_k_distances_idx = np.argpartition(distances,k)[0:k]
            best_k_categories = self.Ytr[best_k_distances_idx]
            print("Best categories", best_k_categories)
            res[test_index] = stats.mode(best_k_categories)[0][0]
        return res


class NearestNeighbourTest(unittest.TestCase):

    def testNearestNeighbour(self):
        Xtr, Ytr, Xte, Yte = datautils.load_CIFAR10('../../data/cifar10/')
        Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
        Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072
        Xte_rows = Xte_rows[0:20,:]
        Yte = Yte[0:20]
        nn = NearestNeighbour()
        nn.train(Xtr_rows, Ytr)
        Yte_predict = nn.predict(Xte_rows)
        print('accuracy: %f' % (np.mean(Yte_predict == Yte)))

    def testKNearestNeightbour(self):
        test_images = 10
        validation_accuracies = []
        Xtr, Ytr, Xte, Yte = datautils.load_CIFAR10('../../data/cifar10/')
        Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
        Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072
        Xte_rows = Xte_rows[0:test_images,:]
        Yte = Yte[0:test_images]
        nn = KNearestNeighbour()
        nn.train(Xtr_rows, Ytr)
        for k in [1,3,5,7,10,20,50,10]:
            Yte_predict = nn.predict(Xte_rows,k)
            acc = np.mean(Yte_predict == Yte)
            print('accuracy: %f' % acc)
            validation_accuracies.append(acc)
        plt.bar(validation_accuracies)
