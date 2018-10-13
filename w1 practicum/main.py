import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


pd.options.display.float_format = '{:.4f}'.format

url = 'https://raw.githubusercontent.com/dwhitena/corporate-training/master/10-week/week1/practicum/diabetes.csv'
data_bytes = requests.get(url).content
data_string = data_bytes.decode('utf-8')

# print(data)

diabetes_df = pd.read_csv(io.StringIO(data_string))

print(diabetes_df.head())
print(diabetes_df.describe())

diabetes_df.hist()
pd.plotting.scatter_matrix(diabetes_df)
# plt.show()

## -- Attempt 1: split array. Bad idea because data might not be in random order.
##total_rows = diabetes_df['y'].count()
##training_rows = int(total_rows * 0.8)
##[training_df, test_df] = np.split(diabetes_df, [training_rows])
##
##real_training_rows = int(training_rows * 0.8)
##[training_df, holdout_df] = np.split(training_df, [real_training_rows])

test_df = diabetes_df.sample(frac = 0.2)
remainder_df = diabetes_df.drop(test_df.index)
holdout_df = remainder_df.sample(frac = 0.2)
training_df = remainder_df.drop(holdout_df.index)

training_df.to_csv('..\\data\\training.csv')
test_df.to_csv('..\\data\\test.csv')
holdout_df.to_csv('..\\data\\holdout.csv')

print('training:\r\n', training_df.describe())
print('test:\r\n', test_df.describe())
print('holdout:\r\n', holdout_df.describe())
