import time
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split

from utils.DataPreprocess import standardize_data

MODEL = ["LinearRegression", "Ridge", "Lasso", "Lasso"]
ridge_params = {
    "alphas" : [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60],
    "cv" : 5
}
elastic_params = {"l1_ratio" : [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                  "alphas" : [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006,0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                  "max_iter": 50000,
                  "cv" :5
                  }
class LRModel(object):
    def __init__(self, model):
        # a random number generator for reproducibility
        self.rng = np.random.default_rng(seed=0)

        if model == 0:
            self.model = LinearRegression()
        if model == 1:
            self.model = RidgeCV()
        if model == 2:
            self.model = LassoCV()
        if model == 3:
            self.model = ElasticNetCV()

    def fitRidge(self, train_x, train_y):
        self.model.set_params(ridge_params)
        train_x = standardize_data(train_x)
        self.model.fit(train_x, train_y)
        alpha = self.model.alpha_
        ridge_params["alphas"] = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4]

    def fitLasso(self, train_x, train_y):
        pass

    def fitElastic(self, train_x, train_y):

        self.model.set_params(elastic_params)
        train_x = standardize_data(train_x)
        self.model.fit(train_x, train_y)
        alpha = self.model.alpha_
        ratio = self.model.l1_ratio_

        # try again for more precision
        elastic_params["l1_ratio"] = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15]
        elastic_params["alphas"] = [alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1,
                                      alpha * 1.15]

    def fit_model(self, train_x, train_y):
        """
        Fit the model on the given training data
        :param train_x: Training featrues as a 2d Numpy array of shape(NUM_SAMPLES,NUM_FEATURES)
        :param train_y: Training canopy height as a 1d Numpy array of shape(NUM_SAMPLESM,)
        """
        train_x = standardize_data(train_x)
        self.model.fit(train_x, train_y)


    def predict(self, x):
        """
        Predict the canopy height for a given set of images
        :param x:
        :return:
        """
        x = standardize_data(x)

    def evaluation(self, y_gt, y_pred):
        """
        Evaluate on the unseen test region, by measuring the mean absolute error (MAE) and the root mean square error (RMSE).
        :param y_pred:
        :param y_gt:
        :return:
        """
        MAE = metrics.mean_absolute_error(y_true=y_gt, y_pred=y_pred)
        RMSE = metrics.mean_squared_error(y_true=y_gt, y_pred=y_pred, squared=False)
        return MAE, RMSE


def main():
    print("Start fitting the model...")
    start = time.time()
    print(f"Finish fitting in {time.time() - start} seconds.")

if __name__ == "__main__":
    main()