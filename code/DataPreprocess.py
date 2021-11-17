from sklearn.preprocessing import StandardScaler


def standardize_data(x):
    '''
    Transform x into unit variance and zero mean
    :param x: input data
    :return:
    '''
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    return x_scaled
