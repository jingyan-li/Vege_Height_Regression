# II_Lab2
Image Interpretation Lab 2 Regression


## To Run Jupyter Notebook
[Check this link](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084)


## Load Environment
Open conda prompt and go the project directory. Then use `conda env create -f env.yml`.


## Code Structure

./preprocess_evaluation -- data preprocess and result evaluation

./utils -- dataset for pre-trained network and shallow predictor

KNN.py

GaussianProcess.py

SGDRegressor.py

XGBoost.py -- shallow predictor

predict.py -- make predictions on test dataset

