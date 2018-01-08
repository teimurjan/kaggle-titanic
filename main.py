import pandas
import lr.lr as lr
import rf.rf as rf

if __name__ == '__main__':
  train_data = pandas.read_csv('data/train.csv', index_col='PassengerId')
  test_data = pandas.read_csv('data/test.csv', index_col='PassengerId')
  lr.analyze_with_plot(train_data)
  rf.analyze_with_plot(train_data)
