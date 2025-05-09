# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10 Part2. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256

        # TODO Define the network architecture (layers) based on these specifications.
       
        # CNN layers
        h = 128

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) 

        self.fc1 = nn.Linear(32 * 7 * 7, h)  # Fully connected layer with 128 hidden units
        self.fc2 = nn.Linear(h, out_size)  # Output layer (out_size classes)

        self.optimizer = optim.SGD(self.parameters(), lr=lrate)  

        self.activation = nn.ReLU()  # You can adjust the negative_slope

        self.pool = nn.MaxPool2d(2, 2) 


    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        # TODO Implement the forward pass.

        batch_size, input_size = x.shape

        reshaped_x = x.view(batch_size, 3, 31, 31)
        reshaped_x = self.pool(self.activation(self.conv1(reshaped_x)))
        reshaped_x = self.pool(self.activation(self.conv2(reshaped_x)))
        reshaped_x = reshaped_x.view(x.size(0), -1)         # Flatten to (batch_size, 16 * 31 * 31)

        # Step 4: Fully connected layers
        
        reshaped_x = self.activation(self.fc1(reshaped_x))  # Apply ReLU activation
        reshaped_x = self.fc2(reshaped_x)                   # Output layer

        return reshaped_x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
    
        # Important, detach and move to cpu before converting to numpy and then to python float.
        # Or just use .item() to convert to python float. It will automatically detach and move to cpu.

        self.optimizer.zero_grad()  # Reset gradients
        yhat = self.forward(x)  # Forward pass
        loss_value = self.loss_fn(yhat, y)  # Calculate the loss
        loss_value.backward()  # Compute gradients
        self.optimizer.step()  # Update weights

        return loss_value.item()



def fit(train_set, train_labels, dev_set, epochs, batch_size = 100):
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """
    # Important, don't forget to detach losses and model predictions and convert them to the right return types.

    mean, std = train_set.mean(dim=0), train_set.std(dim=0)
    
    train_set = (train_set - mean) / std
    dev_set = (dev_set - mean) / std      

    net = NeuralNet(lrate=0.1, loss_fn=nn.CrossEntropyLoss(), in_size=train_set.shape[1], out_size=4)

    train_data = get_dataset_from_arrays(train_set, train_labels)
    train_loader = DataLoader(train_data, batch_size, shuffle=False)

    # train the network
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data['features'], data['labels']
            loss = net.step(inputs, labels)
            running_loss += loss


        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

    dev_set_tensor = torch.tensor(dev_set).float()
    dev_preds = net(dev_set_tensor)
    predicted_labels = np.argmax(dev_preds.detach().numpy(), axis=1)

    return losses, predicted_labels, net