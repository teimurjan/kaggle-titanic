import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame


def plot(x_labels, scores):
  plt.plot(x_labels, scores)
  plt.xlabel('parameter')
  best_score = max(scores)
  best_x = x_labels[scores.index(best_score)]
  plt.scatter(best_x, best_score, marker='o', c='r')
  plt.annotate('Best score is {:0.3f}\n when parameter is {}'.format(best_score, best_x), (best_x, best_score))
  plt.ylabel('score')
  plt.show()


def prepare_X_train(data, scaler=None):
  X = data[['Pclass', 'Age', 'Fare', 'Sex']]
  X['Sex'] = np.where(X['Sex'] == 'male', 1, 0)
  X['Age'].fillna(data['Age'].mean(), inplace=True)
  X['Family'] = data['SibSp'] + data['Parch']
  return scaler.fit_transform(X) if scaler else X


def prepare_X_test(data, scaler=None):
  X = data[['Pclass', 'Age', 'Fare', 'Sex']]
  X['Sex'] = np.where(X['Sex'] == 'male', 1, 0)
  X['Fare'].fillna(data['Fare'].median(), inplace=True)
  X['Age'].fillna(data['Age'].mean(), inplace=True)
  X['Family'] = data['SibSp'] + data['Parch']
  return scaler.transform(X) if scaler else X


def save_result(predicted, index_col):
  result = DataFrame({'Survived': predicted}, index=index_col)
  result.to_csv('gender_submission.csv')
