import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier

useful_cols = ['u', 'g', 'r', 'i', 'z', 'run', 'camcol', 'field', 'class', 'plate', 'mjd', 'fiberid', 'x_coord', 'y_coord', 'z_coord']


def loadData():
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")

    test.dropna(inplace=True)
    train.dropna(inplace=True)

    encoder_train = LabelEncoder()
    train['class'] = encoder_train.fit_transform(train['class'])

    encoder_test = LabelEncoder()
    test['class'] = encoder_test.fit_transform(test['class'])

    return test[useful_cols], train[useful_cols]


def boostingClassifier(train, test):
    print("-------------------------BOOSTING-------------------------")
    y_train = train['class']
    X_train = train.drop(['class'], axis=1)

    y_test = test['class']
    X_test = test.drop(['class'], axis=1)

    ada_boost = AdaBoostClassifier(n_estimators=550, algorithm='SAMME')
    ada_boost.fit(X_train, y_train)
    pred = ada_boost.predict(X_test)
    print(confusion_matrix(y_test.to_numpy(), pred))
    print(ada_boost.score(X_test, y_test))

def baggingClassifier(train, test):
    print("\n--------------------------BAGGING-------------------------")
    y_train = train['class']
    X_train = train.drop(['class'], axis=1)

    y_test = test['class']
    X_test = test.drop(['class'], axis=1)

    bag = BaggingClassifier(base_estimator=SVC(), n_estimators=20)
    bag.fit(X_train, y_train)
    pred = bag.predict(X_test)
    print(confusion_matrix(y_test.to_numpy(), pred))
    print(bag.score(X_test, y_test))

def randomForest(train, test):
    print("\n--------------------------RANDOM FOREST-------------------------")
    y_train = train['class']
    X_train = train.drop(['class'], axis=1)

    y_test = test['class']
    X_test = test.drop(['class'], axis=1)

    forest = RandomForestClassifier(max_depth=50)
    forest.fit(X_train, y_train)
    pred = forest.predict(X_test)
    print(confusion_matrix(y_test.to_numpy(), pred))
    print(forest.score(X_test, y_test))

    estimator = forest.estimators_[5]

    from sklearn.tree import export_graphviz
    # Export as dot file
    export_graphviz(estimator,
                    out_file="tree.dot",
                    filled=True,
                    rounded=True)

    os.system('dot -Tpng tree.dot -o tree.png')

def stackingClassifier(train, test):
    print("\n--------------------------STACKING-------------------------")
    y_train = train['class']
    X_train = train.drop(['class'], axis=1)

    y_test = test['class']
    X_test = test.drop(['class'], axis=1)


    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svr', make_pipeline(StandardScaler(), LinearSVC()))
    ]

    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    stack.fit(X_train, y_train)
    pred = stack.predict(X_test)
    print(confusion_matrix(y_test.to_numpy(), pred))
    print(stack.score(X_test, y_test))

def dicisionTreeRegressor(train, test):
    print("\n--------------------------DECISION TREE REGRESSION-------------------------")
    y_train = train[['x_coord','y_coord','z_coord']]
    X_train = train.drop(['class','x_coord','y_coord','z_coord'], axis=1)

    y_test = test[['x_coord','y_coord','z_coord']]
    X_test = test.drop(['class','x_coord','y_coord','z_coord'], axis=1)


    model = DecisionTreeRegressor(criterion='mse', max_features='auto')

    model.fit(X_train, y_train)
    model.predict(X_test)
    print(model.score(X_test, y_test))

def linearReggression(train, test):
    print("\n--------------------------LINEAR REGRESSION-------------------------")
    y_train = train[['x_coord', 'y_coord', 'z_coord']]
    X_train = train.drop(['class', 'x_coord', 'y_coord', 'z_coord'], axis=1)

    y_test = test[['x_coord', 'y_coord', 'z_coord']]
    X_test = test.drop(['class', 'x_coord', 'y_coord', 'z_coord'], axis=1)

    model = LinearRegression()
    model.fit(X_train, y_train)

    model.predict(X_test)
    print(model.score(X_test, y_test))

test, train = loadData()

linearReggression(train, test)
boostingClassifier(train, test)
baggingClassifier(train, test)
randomForest(train, test)
stackingClassifier(train, test)
dicisionTreeRegressor(train, test)
