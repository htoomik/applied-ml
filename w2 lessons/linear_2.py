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
# learning_rate: learning rate for SGD
# epochs - number of epochs for SGD loop
# lambda1 - L1 regularization rate
# lambda2 - L2 regularization rate
def pytorch_lr_fit(x, y, learning_rate, epochs, lambda1, lambda2):

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

    # L2 regularization is built in
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = lambda2)
    
    for epoch in range(epochs):
        features = Variable(torch.from_numpy(x).float(), requires_grad = True)
        response = Variable(torch.from_numpy(y).float())

        # the optimizer remembers gradients from the previous iteration - reset
        optimizer.zero_grad()

        predictions = model.forward(features)

        loss = loss_func(predictions, response)

        # L1 regularization needs to be done "manually"
        if lambda1 > 0.0:
            # view(-1) flattens each parameter
            # cat concatenates into a single list/array/whatever
            parameters = torch.cat([x.view(-1) for x in model.linear.parameters()])
            l1_regularization = lambda1 * torch.norm(parameters, 1)
            loss += l1_regularization

        # calculate the derivative/gradient for each feature
        loss.backward()

        # based on the gradients, take a step in the right direction
        optimizer.step()

    return model

def main():
    scaler = MinMaxScaler()
    columns = ['bmi', 'map', 'ldl', 'hdl', 'tch', 'glu', 'ltg', 'y']
    feature_count = len(columns[0:-1])
    
    # training data
    training_raw = pd.read_csv('..\\data\\training.csv')
    
    # fit_transform: fits the scaler to the data set, and transforms data set
    training_scaled = scaler.fit_transform(training_raw[columns])

    training_features = np.array(training_scaled[:, 0:feature_count])
    training_response = np.array(training_scaled[:, feature_count])

    # train model
    model_linear = pytorch_lr_fit(training_features, training_response, 0.1, 1000, 0.0, 0.0)
    model_lasso = pytorch_lr_fit(training_features, training_response, 0.1, 1000, 0.001, 0.0)
    model_ridge = pytorch_lr_fit(training_features, training_response, 0.1, 1000, 0.0, 0.001)
    model_enet = pytorch_lr_fit(training_features, training_response, 0.1, 1000, 0.001, 0.001)

    # testing data
    test_raw = pd.read_csv('..\\data\\test.csv')
    
    # transform: reuses already-fitted scaler, so that test data is scaled
    # using the same scale as the training data, for apples-to-apples comparison
    test_scaled = scaler.transform(test_raw[columns])
    test_features = test_scaled[:, 0:feature_count]
    test_response = test_scaled[:, feature_count]

    test_input = Variable(torch.from_numpy(test_features).float())

    # test all the models
    predictions_linear = model_linear(test_input)
    predictions_lasso = model_lasso(test_input)
    predictions_ridge = model_ridge(test_input)
    predictions_enet = model_enet(test_input)

    # evaluate test results
    rmse_linear = math.sqrt(mean_squared_error(predictions_linear.data.numpy(), test_response))
    rmse_lasso = math.sqrt(mean_squared_error(predictions_lasso.data.numpy(), test_response))
    rmse_ridge = math.sqrt(mean_squared_error(predictions_ridge.data.numpy(), test_response))
    rmse_enet = math.sqrt(mean_squared_error(predictions_enet.data.numpy(), test_response))
    
    print('RMSE no regularization: %0.4f'% rmse_linear)
    print('RMSE lasso: %0.4f'% rmse_lasso)
    print('RMSE ridge: %0.4f'% rmse_ridge)
    print('RMSE elastic net: %0.4f'% rmse_enet)
    

if __name__ == '__main__':
    main()
