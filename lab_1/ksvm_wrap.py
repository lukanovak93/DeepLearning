from sklearn import svm
import numpy as np
import data

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.clf = svm.SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.clf.fit(X, Y_)

    def predict(self, X):
        return self.clf.predict(X)

    def get_scores(self, X):
        return self.clf.decision_function(X)

    def support(self):
        return self.clf.support_

if __name__ == "__main__":
    np.random.seed(100)
    (X, Y_) = data.sample_gmm_2d(6, 2, 10)
    ksvm = KSVMWrap(X, Y_)
    Y = ksvm.predict(X)
    acc, prec, rec, avg_prec = data.eval_perf_binary(Y, Y_)
    print("Accuracy: {}\nPrecision: {}\nRecall: {}\nAverage Precision: {}".format(acc, prec, rec, avg_prec))
    data.graph_surface(ksvm.get_scores, (np.min(X, axis=0), np.max(X, axis=0)))
    data.graph_data(X, Y_, Y, special=ksvm.support())
