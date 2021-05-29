import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def numerize(string):
    if string is None:
        return -1

    if string == "normal":
        return 0
    elif string == "above normal":
        return 0.5
    elif string == "well above normal":
        return 1
    elif string == "man":
        return 0
    elif string == "woman":
        return 1


def normalize(tmp):
    age = tmp['age'] / 365
    height = tmp['height']
    weight = tmp['weight']
    ap_hi = tmp['ap_lo']
    ap_lo = tmp['ap_hi']
    if age < 10 or age > 110:
        return 0
    if height < 140 or height > 220:
        return 0
    if weight < 40 or weight > 180:
        return 0
    if ap_hi < 50 or ap_hi > 200:
        return 0
    if ap_lo < 40 or ap_lo > 200:
        return 0
    return 1


def addBMI(data):
    data.insert(0, value=float, column="bmi")

    for index, row in data.iterrows():
        data.at[index, 'bmi'] = row['weight'] / (row['height'] ** 2)
    return data


def getAndPreProcessData():
    with open('srdcove_choroby.csv', newline='') as data:
        dataSet = pd.read_csv(data)
        dataSet.dropna(inplace=True)
        dataSet.drop(["id"], axis=1, inplace=True)
        dataSet.dropna(inplace=True)

        dataSet = addBMI(dataSet)

        for index, row in dataSet.iterrows():
            if normalize(row) == 0:
                dataSet.drop(index, inplace=True)
                continue

            dataSet.at[index, 'age'] = row['age'] / 365

            x = numerize(row['cholesterol'])
            if x is not None and x >= 0:
                dataSet.at[index, 'cholesterol'] = x
            else:
                dataSet.drop(index, inplace=True)
                continue

            x = numerize(row['glucose'])
            if x >= 0:
                dataSet.at[index, 'glucose'] = x
            else:
                dataSet.drop(index, inplace=True)
                continue

            x = numerize(row['gender'])
            if x >= 0:
                dataSet.at[index, 'gender'] = x
            else:
                dataSet.drop(index, inplace=True)
                continue
            print(index)

        dataSet.to_csv('tip_top.csv')


def removeFromList(lst, remove):
    for x in remove:
        lst.remove(x)
    return lst


def trainBinaryClassificator(data):
    lst = list(data.columns)
    lst.remove('cardio')
    lst.remove('bmi')

    X = data[lst]
    y = data['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, learning_rate='adaptive', verbose=True)

    mlp.fit(X_train, y_train.values.ravel())
    predictions = mlp.predict(X_test)

    print(classification_report(y_test, predictions))
    print('Mean Squared Error:', mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', r2_score(y_test, predictions))


def trainRegressor(data):
    lst = list(data.columns)
    lst.remove('bmi')
    lst.remove('weight')
    lst.remove('height')

    X = data[lst]
    y = data['bmi']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("---------------MLPRegressor-----------------")
    nn_regressor = MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=1000, learning_rate='adaptive')
    nn_regressor.fit(X_train, y_train)
    y_pred_nn = nn_regressor.predict(X_test)

    print('Mean Squared Error:', mean_squared_error(y_test, y_pred_nn))
    print('Root Mean Squared Error:', r2_score(y_test, y_pred_nn))

    print("---------------LinearRegressor-----------------")
    lin_regressor = LinearRegression()
    lin_regressor.fit(X_train, y_train)
    y_pred_lin = lin_regressor.predict(X_test)

    print('Mean Squared Error:', mean_squared_error(y_test, y_pred_lin))
    print('Root Mean Squared Error:', r2_score(y_test, y_pred_lin))


try:
    data = pd.read_csv('tip_top.csv')
except FileNotFoundError:
    getAndPreProcessData()
    data = pd.read_csv('tip_top.csv')
finally:
    data.drop(data.columns[0], axis=1, inplace=True)


trainBinaryClassificator(data)
print("---------------------------------------------------------------------------")
trainRegressor(data)
