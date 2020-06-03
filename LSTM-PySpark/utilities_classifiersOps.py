def preprocessing_Multiclass(df, labelName, dismissCols=[], dummies=False, scale=False, scaler=None, labelEncoder=None):
    """
    Separate features and encoded labels from DataFrame.
    Arguments:
        df (Pandas DataFrame): DataFrame to preprocess.
        labelName (string): Labels as Numpy array.
        dismissCols (list): List of columns to dismiss.
        dummies (bool): If True encodes the classes as dichotomous variables.
        scale (bool): If True scale the features with MinMaxScaller
    Returns:
        features (array): Features array
        labels (array): Labels array.
        labelEncoder (LabelEncoder object): Label encoder fitted on df.
    """
    from numpy import ravel
    from sklearn.preprocessing import LabelEncoder
    from pandas import get_dummies
    from sklearn.preprocessing import MinMaxScaler

    if not scale:
        features = df.drop(dismissCols+labelName, axis=1).values
    else:
        if scaler is None:
            scaler = MinMaxScaler()
        features = scaler.fit_transform(df.drop(dismissCols+labelName, axis=1).values)

    if not dummies:
        if labelEncoder is None:
            labelEncoder = LabelEncoder()
        labels = labelEncoder.fit_transform(ravel(df[labelName]))

    else:
        labels = get_dummies(df[labelName]).values
        labelColumns = get_dummies(df[labelName]).columns.to_list()

    if not dummies and not scale:
        return features, labels, labelEncoder
    elif not dummies and scale:
        return features, labels, labelEncoder, scaler
    elif dummies and not scale:
        return features, labels, labelColumns
    elif dummies and scale:
        return features, labels, labelColumns, scaler

def XGB_timeSeries_multiClass(features, labels, n_classes, n_split, returnModel=False, resultsAsDict=False):
    """
    Train a XGB model with the inputed data.
    Arguments:
        features (array): Features array.
        labels (array): Labels array.
        n_classes (int): Number of classes to predict.
        n_split (int): Number of splits.
        returnModel (bool): If true returns the best model.
        resultsAsDict (bool): Return the results as dict.
    Returns:
        results (list or dict): List with metrics evaluation for each split.
        cm (list): List of arrays containing confusion matrices from predictions of each split.
        model (booster): Best model based com f1-score.
    """
    from sklearn.model_selection import TimeSeriesSplit
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import confusion_matrix

    # Cross validator object for time series
    kfold = TimeSeriesSplit(n_splits = n_split)
    # List to save results
    results = []
    cm = []
    resultsDict = []
    models = []
    # Xgboost params
    params = {
        'max_depth': 6,
        'objective': 'multi:softmax',  # error evaluation for multiclass training
        'num_class': n_classes,
        'n_gpus': 0
                }

    for train_index, test_index in kfold.split(features, labels):
        dtrain = xgb.DMatrix(data=features[train_index], label=labels[train_index])
        dtest = xgb.DMatrix(data=features[test_index])
        estimator = xgb.train(params, dtrain)
        pred = estimator.predict(dtest)
        models.append(estimator)
        results.append(classification_report(labels[test_index], pred))
        cm.append(confusion_matrix(labels[test_index], pred))
        resultsDict.append(classification_report(labels[test_index], pred, output_dict=True))

    #Getting best model
    f1_list = []
    for result in resultsDict:
        f1_list.append(result['macro avg']['f1-score'])
    bestModelIndex = f1_list.index(max(f1_list))

    if returnModel and not resultsAsDict:
        return results, cm, models[bestModelIndex]
    elif returnModel and resultsAsDict:
        return resultsDict, cm, models[bestModelIndex]
    elif not returnModel and resultsAsDict:
        return resultsDict, cm
    elif not returnModel and not resultsAsDict:
        return results, cm


def plot_confusion_matrix(cm, classes, normalized=True, cmap='bone'):
    """
    Plots confusion matrices.
    Arguments:
        cm (list of arrays): List containing arrays from confusion matrices.
        classes (list of strings): List containing ordered classes names.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 9))
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', linewidths=.5, xticklabels=classes, yticklabels=classes, cmap=cmap, ax=ax)
        b, t = ax.get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax.set_ylim(b, t) # update the ylim(bottom, top) values
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicted')
