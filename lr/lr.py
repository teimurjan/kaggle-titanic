from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from utils import prepare_X_train, plot, prepare_X_test, save_result


def analyze_with_plot(data):
  scaler = StandardScaler()
  X_train = prepare_X_train(data, scaler)
  y_train = data['Survived']

  cv = KFold(n_splits=5, shuffle=True, random_state=241)

  C_pows = range(-5, 5)
  C_range = [10.0 ** i for i in C_pows]
  scores = []
  for C in C_range:
    clf = LogisticRegression(C=C, random_state=241)
    score = cross_val_score(estimator=clf, cv=cv, X=X_train, y=y_train, scoring='roc_auc').mean()
    scores.append(score)
  plot(C_pows, scores)


def predict_result(train_data, test_data):
  scaler = StandardScaler()
  X_train = prepare_X_train(train_data, scaler)
  X_test = prepare_X_test(test_data, scaler)
  y_train = train_data['Survived']
  clf = LogisticRegression(C=1000.0, random_state=241)
  clf.fit(X_train, y_train)
  predicted = clf.predict(X_test)
  save_result(predicted, X_test.index)
