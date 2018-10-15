import math
import torch
import pandas as pd
import numpy as np
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# define a model by inheriting
class PyTorchLRModel(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(PyTorchLRModel, self).__init__()

        # the Linear module applies a linear transformation to incoming data        
        self.linear = torch.nn.Linear(in_size, out_size)
        
    def forward(self, x):
        # forward performs the actual model operation

        out = self.linear(x)
        return out

# Trains a model
# x: features - numpy array
# y: response - numpy array
def pytorch_lr_fit(x, y, learning_rate, epochs):

    # number of dimensions in incoming data
    input_dimension = x.ndim
    output_dimension = y.ndim

    # if the features are a 1-d array (a list), turn into a 2d column vector
    if input_dimension == 1:
        x = x[:, np.newaxis]
        input_size = 1
    else:
        input_size = x.shape[1]

    if output_dimension == 1:
        y = y[:, np.newaxis]
        output_size = 1
    else:
        output_size = y.shape[1]


    model = PyTorchLRModel(input_size, output_size)

    # We will use Mean Square Error as loss function
    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    
    for epoch in range(epochs):
        features = Variable(torch.from_numpy(x).float(), requires_grad = True)
        response = Variable(torch.from_numpy(y).float())

        # the optimizer remembers gradients from the previous iteration - reset
        optimizer.zero_grad()

        predictions = model.forward(features)

        loss = loss_func(predictions, response)

        # calculate the derivative/gradient for each feature
        loss.backward()

        # based on the gradients, take a step in the right direction
        optimizer.step()

    return model

def main():
    # training data
    training_raw = pd.read_csv('..\\data\\training.csv')

    scaler = MinMaxScaler()
    
    # fit_transform: fits the scaler to the data set, and transforms data set
    training_scaled = scaler.fit_transform(training_raw[['bmi', 'ltg', 'y']])

    training_features = np.array(training_scaled[:, 0:2])
    training_response = np.array(training_scaled[:, 2])

    # train model
    model = pytorch_lr_fit(training_features, training_response, 0.1, 1000)

    # testing data
    test_raw = pd.read_csv('..\\data\\test.csv')
    
    # transform: reuses already-fitted scaler, so that test data is scaled
    # using the same scale as the training data, for apples-to-apples comparison
    test_scaled = scaler.transform(test_raw[['bmi', 'ltg', 'y']])
    test_features = test_scaled[:, 0:2]
    test_response = test_scaled[:, 2]

    # test model
    test_input = Variable(torch.from_numpy(test_features).float())
    predictions = model(test_input)

    # evaluate test results
    
    rmse = math.sqrt(mean_squared_error(predictions.data.numpy(), test_response))
    print('RMSE: %0.4f'% rmse)
    

if __name__ == '__main__':
    main()
