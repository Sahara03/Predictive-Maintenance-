from copy import deepcopy
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
import datetime
import numpy as np

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    colNames = data.columns.to_list()
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(colName+'(t-%d)' % (i)) for colName in colNames]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(colName+'(t)') for colName in colNames]
        else:
            names += [(colName+'(t+%d)' % (i)) for colName in colNames]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# split a multivariate sequence into samples
def split_sequences(features, labels, n_steps):
    """
    Split the feature and labels series in batches.
    Arguments:
            features: Features as NumPy array.
            labels: Labels as Numpy array.
            n_steps: Number o timesteps inside a batch.
        Returns:
            Resampled features and labels array.
    """
    X, y = list(), list()
    assert len(features)==len(labels), "Features and labels lenght must be the same!"
    for i in range(len(features)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(features)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = features[i:end_ix, :], labels[end_ix-1, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def invertScale(batchs_array, scaler):
    """
        Invert the scaled data, can be one or more batches.
        Arguments:
            batchs_array: NumPy array to be inverted.
            scaler: Scaler object fitted into the original data.
        Returns:
            Numpy array with the inverted values.
    """
    if len(batchs_array.shape) == 3:
        inverted = []
        for i in range(batchs_array.shape[1]):
            inverted_batch = scaler.inverse_transform(batchs_array[:,i,:])
            inverted.append(inverted_batch)
        return np.array(inverted).reshape(batchs_array.shape[0], batchs_array.shape[1], batchs_array.shape[2])
    else:
        return scaler.inverse_transform(batchs_array)

def multvariative_multstep_dataPrep(df, n_past, n_future, look_back, return_df=False):
    """
    Frame a time series as a supervised learning dataset and sample them.
    Arguments:
        df: Pandas timeseries df
        n_past: Number of lag observations as input (X).
        n_future: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Features and Labels arrays, scalers and dataset.
    """
    scalerX = MinMaxScaler(feature_range=(0,1))
    scalerY = MinMaxScaler(feature_range=(0,1))
    dataset = series_to_supervised(df, n_in=n_past, n_out=n_future)
    x = dataset.iloc[:,~dataset.columns.str.contains('t+', regex=False)].values
    y = dataset.iloc[:,dataset.columns.str.contains('t+', regex=False)].values
    scaledX = scalerX.fit_transform(x)
    scaledY = scalerY.fit_transform(y)
    feat, label = split_sequences(scaledX, scaledY, look_back)
    if return_df:
        return feat, label, scalerX, scalerY, dataset
    else:
        return feat, label, scalerX, scalerY

def train_model_renom(model, x, y, testSize, max_epoch, batch_size, period, optimizer, saveFig=False, figName=None):
    """
    Train a LSTM renom model with early stopping.
    Arguments:
        model: renom sequential model.
        x: fetures array.
        y: labels array.
        testSize: float test size.
        max_epoch: int
        batch_size: float
        .
        .
        .
    Returns:
        fitted model, learn loss values and test loss values
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testSize)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('train size: {}, test size: {}'.format(train_size, test_size))

    # Train Loop
    epoch = 0
    loss_prev = np.inf

    learning_curve, test_curve = [], []

    while(epoch < max_epoch):
        epoch += 1

        perm = np.random.permutation(train_size)
        train_loss = 0

        for i in range(train_size // batch_size):
            batch_x = X_train[perm[i*batch_size:(i+1)*batch_size]]
            batch_y = y_train[perm[i*batch_size:(i+1)*batch_size]]

            # Forward propagation
            l = 0
            z = 0
            with model.train():
                for t in range(look_back):
                    z = model(batch_x[:,t])
                    l = rm.mse(z, batch_y)
                model.truncate()
            l.grad().update(optimizer)
            train_loss += l.as_ndarray()

        train_loss /= (train_size // batch_size)
        learning_curve.append(train_loss)

        # test
        l = 0
        z = 0
        for t in range(look_back):
            z = model(X_test[:,t])
            l = rm.mse(z, y_test)
        model.truncate()
        test_loss = l.as_ndarray()
        test_curve.append(test_loss)

        # check early stopping
        if epoch % period == 0:
            print('epoch:{} train loss:{} test loss:{}'.format(epoch, train_loss, test_loss))
            if test_loss > loss_prev*0.99:
                print('Stop learning')
                break
            else:
                loss_prev = deepcopy(test_loss)

    plt.figure(figsize=(10,5))
    plt.plot(learning_curve, color='b', label='learning curve')
    plt.plot(test_curve, color='orange', label='test curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(fontsize=20)
    if saveFig:
        plt.savefig(figName+'.png')

    plt.show()
    return model, learning_curve[-1], test_curve[-1]

# calculate Mahalanobis distance
def Mahala_distantce(x,mean,cov):
    d = np.dot(x-mean,np.linalg.inv(cov))
    d = np.dot(d, (x-mean).T)
    return d

def fit_lstm(feat, label, testSize, n_neurons, loss, optimizer, activation, n_epoch):

    model = Sequential()
    model.add(LSTM(n_neurons, activation='relu', input_shape=(feat.shape[1], feat.shape[2])))
    model.add(Dense(label.shape[1]))
    model.compile(loss=loss, optimizer=optimizer)
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    history = model.fit(feat, label, epochs=n_epoch, verbose=2, validation_split=testSize, \
                        shuffle=False, callbacks=[es])
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    return model

def make_predictions(model, features, scaler):
    preds = []
    for t in range(features.shape[0]):
        pred = invertScale(model.predict(features[t].reshape(1,features[-1].shape[0],features[-1].shape[1])),\
                                   scaler)
        preds.append(pred)
    preds = np.array(preds)
    preds = preds.reshape(preds.shape[0],preds.shape[2])
    return preds

def make_df_pred(original_df, preds, look_back):
    df_pred = DataFrame(columns=original_df.columns.to_list(), index=original_df.index)

    df_pred.iloc[:, ~df_pred.columns.str.contains('t+', regex=False)] = original_df\
                                        .iloc[:, ~original_df.columns.str.contains('t+', regex=False)].values

    df_pred.iloc[:df_pred.shape[0]-look_back, df_pred.columns.str.contains('t+', regex=False)] = preds
    df_pred = df_pred.dropna(how='any')
    return df_pred

def plot_predictions_forecasts(df_pred, col, n_future, timeStep, metrica, figSize):
    figSize = (20,6)
    df_real = df_pred.iloc[:,(df_pred.columns.str.contains('(t)', regex=False))&(df_pred.columns.str.contains(col))]
    df_real = df_real.rename({df_real.columns.to_list()[0]: col}, axis=1)
    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(df_real.index, df_real[col], label='Real values', color='blue')
    for i in range(len(df_pred)):
        start = df_pred.index[i]
        date_list = [start + datetime.timedelta(minutes=timeStep*x) for x in range(n_future)]
        pred_series = df_pred.iloc[i,(~df_pred.columns.str.contains('t-', regex=False))&(df_pred.columns.str.contains(col))]
        if i==0:
            ax.plot(date_list, pred_series, label='Predicted values', color='red')
        else:
            ax.plot(date_list, pred_series, color='red')
    #ax.axvline(x=datetime.datetime(2019,10,25, 6,58,0), linewidth=1, color='black', ls='dashed', label='Incident')
    ax.set_title(metrica+': '+col)
    ax.set_ylabel('%')
    ax.set_xlabel('date')
    ax.legend()
    return fig

def plot_nth_prediction(df_pred, col, n_th, timeStep, metrica, figSize):
    df_real = df_pred.iloc[:,(df_pred.columns.str.contains('(t)', regex=False))&(df_pred.columns.str.contains(col))]
    df_real = df_real.rename({df_real.columns.to_list()[0]: col}, axis=1)
    df_pred1 = df_pred.iloc[:,(df_pred.columns.str.contains('(t+'+str(n_th)+')', regex=False))&(df_pred.columns.str.contains(col))]
    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(df_real.index, df_real[col], label='Real values', color='blue')
    ax.plot([date+datetime.timedelta(minutes=n_th*timeStep) for date in df_pred1.index], df_pred1, label='Pred values', color='red')
    ax.set_title(metrica+': '+col)
    ax.set_ylabel('%')
    ax.set_xlabel('date')
    ax.legend()
    return fig
