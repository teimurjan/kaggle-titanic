from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

from utils import prepare_X_train, plot, prepare_X_test, save_result


def analyze_with_plot(data):
  X_train = prepare_X_train(data)
  y_train = data['Survived']
  cv = KFold(n_splits=5, shuffle=True, random_state=241)
  scores = []
  n_estimators = [100, 200, 300]
  for n in n_estimators:
    clf = RandomForestClassifier(n_estimators=n, random_state=241)
    score = cross_val_score(estimator=clf, cv=cv, X=X_train, y=y_train, scoring='roc_auc').mean()
    scores.append(score)
  plot(n_estimators, scores)


def predict_result(train_data, test_data):
  X_train = prepare_X_train(train_data)
  X_test = prepare_X_test(test_data)
  y_train = train_data['Survived']
  clf = RandomForestClassifier(n_estimators=200, random_state=241)
  clf.fit(X_train, y_train)
  predicted = clf.predict(X_test)
  save_result(predicted)
