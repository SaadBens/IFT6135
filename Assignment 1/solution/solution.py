# This script contains the helper functions you will be using for this assignment

import os
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BassetDataset(Dataset):
    """
    BassetDataset class taken with permission from Dr. Ahmad Pesaranghader

    We have already processed the data in HDF5 format: er.h5
    See https://www.h5py.org/ for details of the Python package used

    We used the same data processing pipeline as the paper.
    You can find the code here: https://github.com/davek44/Basset
    """

    # Initializes the BassetDataset
    def __init__(self, path='./data/', f5name='er.h5', split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.split = split

        split_dict = {'train': ['train_in', 'train_out'],
                      'test': ['test_in', 'test_out'],
                      'valid': ['valid_in', 'valid_out']}

        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"

        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')

        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']

        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]

        self.ids = list(range(len(self.inputs)))
        if self.split == 'test':
            self.id_vars = np.char.decode(self.dataset['test_headers'])

    def __getitem__(self, i):
        """
        Returns the sequence and the target at index i

        Notes:
        * The data is stored as float16, however, your model will expect float32.
          Do the type conversion here!
        * Pay attention to the output shape of the data.
          Change it to match what the model is expecting
          hint: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        * The target must also be converted to float32
        * When in doubt, look at the output of __getitem__ !
        """

        idx = self.ids[i]

        # Sequence & Target
        output = {'sequence': None, 'target': None}

        # WRITE CODE HERE
        output["sequence"] = torch.tensor(self.inputs[idx], dtype=torch.float32)
        output["sequence"] = output["sequence"].permute(1,2,0)
        output["target"] = torch.tensor(self.outputs[idx], dtype=torch.float32)

        return output

    def __len__(self):
        # WRITE CODE HERE
        return len(self.inputs)

    def get_seq_len(self):
        """
        Answer to Q1 part 2
        """
        # WRITE CODE HERE
        output = self.__getitem__(0)["sequence"].size()[1]
        return output

    def is_equivalent(self):
        """
        Answer to Q1 part 3
        """
        # WRITE CODE HERE
        output = self.__getitem__(0)
        if output["sequence"].size() == torch.Size([1, self.get_seq_len(), 4]) :
          resp = True
        else :
          resp = False
        return resp


class Basset(nn.Module):
    """
    Basset model
    Architecture specifications can be found in the supplementary material
    You will also need to use some Convolution Arithmetic
    """

    def __init__(self):
        super(Basset, self).__init__()

        self.dropout = .3  # should be float
        self.num_cell_types = 164

        self.dropoutt = nn.Dropout(self.dropout)
        self.conv1 = nn.Conv2d(1, 300, (19, 4), stride=(1, 1), padding=(9, 0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride=(1, 1), padding=(5, 0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))

        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(13*200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)

    def forward(self, x):
        """
        Compute forward pass for the model.
        nn.Module will automatically create the `.backward` method!

        Note:
            * You will have to use torch's functional interface to 
              complete the forward method as it appears in the supplementary material
            * There are additional batch norm layers defined in `__init__`
              which you will want to use on your fully connected layers
        """

        # WRITE CODE HERE
        # Pass the input through the first layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        # Pass the input through the second layer
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        # Pass the input through the third layer
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)
        x = x.reshape(x.size(0), -1)
        # Pass the input through the next layer
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropoutt(x)
        # Pass the input through the next layer
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropoutt(x)
        # output
        x = self.fc3(x)
        return x


def compute_fpr_tpr(y_true, y_pred):
    """
    Computes the False Positive Rate and True Positive Rate
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_pred: model decisions (np.array of ints [0 or 1])

    :Return: dict with tpr, fpr (values are floats)
    """
    output = {'fpr': 0., 'tpr': 0.}

    # WRITE CODE HERE
    TPR = 0
    FPR = 0
    # Calculate true positive and false positive
    positive = np.sum(y_true == 1)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    output['tpr'] = np.float(tp/(tp + fn))
    output['fpr'] = np.float(fp/(fp + tn))

    return output


def compute_fpr_tpr_dumb_model():
    """
    Simulates a dumb model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds (k)
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05
                 ...
                 output['fpr_list'][-1] corresponds to k=0.95

            Do the same for output['tpr_list']
             
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # WRITE CODE HERE
    # Generate random binary target
    target = np.random.randint(2, size=1000)
    # Generate random variables between 0 and 1
    variables = np.random.uniform(low=0, high=1, size=(1000,))
    # List with k values
    K = np.arange(0.0, 1.0, 0.05)
    # Calculate dumb predictions for each K
    for k in K :
      dumb_pred = np.zeros(1000)
      dumb_pred[variables >= k] = 1
      output_k = compute_fpr_tpr(target, dumb_pred)
      output['fpr_list'].append(output_k['fpr'])
      output['tpr_list'].append(output_k['tpr'])
    
    return output


def compute_fpr_tpr_smart_model():
    """
    Simulates a smart model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds (k)
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05
                 ...
                 output['fpr_list'][-1] corresponds to k=0.95

            Do the same for output['tpr_list']
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # WRITE CODE HERE
    # Generate random binary target
    target = np.random.randint(2, size=1000)
    # Generate random variables between 0 and 1
    variables = np.zeros(1000)
    for i in range(len(target)) :
      if target[i] == 0 :
        variables[i] = random.uniform(.0, 0.6)
      else :
        variables[i] = random.uniform(0.4, 1)
    # List with k values
    K = np.arange(0.0, 1.0, 0.05)
    # Calculate dumb predictions for each K
    for k in K :
      dumb_pred = np.zeros(1000)
      dumb_pred[variables >= k] = 1
      output_k = compute_fpr_tpr(target, dumb_pred)
      output['fpr_list'].append(output_k['fpr'])
      output['tpr_list'].append(output_k['tpr'])
    
    return output


def compute_auc_both_models():
    """
    Simulates a dumb model and a smart model and computes the AUC of both

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values in the lists should be floats
    """
    output = {'auc_dumb_model': 0., 'auc_smart_model': 0.}

    # WRITE CODE HERE
    # Generate random binary target
    target = np.random.randint(2, size=1000)
    # Generate random variables between 0 and 1
    dumb = np.random.uniform(low=0, high=1, size=(1000,))
    smart = np.zeros(1000)
    for i in range(len(target)) :
      if target[i] == 0 :
        smart[i] = random.uniform(.0, 0.6)
      else :
        smart[i] = random.uniform(0.4, 1)
    
    output['auc_dumb_model'] = compute_auc(target, dumb)['auc']
    output['auc_smart_model'] = compute_auc(target, smart)['auc']

    return output


def compute_auc_untrained_model(model, dataloader, device):
    """
    Computes the AUC of your input model

    Dont forget to re-apply your output activation!

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values should be floats

    Make sure this function works with arbitrarily small dataset sizes!
    """
    output = {'auc': 0.}

    # WRITE CODE HERE
    model = model.to(device)
    model.eval()
    auc_test = []
    y_model_all = torch.Tensor()
    y_true_all = torch.Tensor()
    
    with torch.set_grad_enabled(False) :
      for i, dictionary in enumerate(dataloader):
        # Transfer to GPU
        sequence, target = dictionary["sequence"].to(device), dictionary["target"].to(device)
        # Model computations
        outputs = model(sequence)
        y_model = torch.sigmoid(outputs)
        # # Concatenate y_true
        y_true = target.cpu()
        y_true_all = torch.cat((y_true_all, y_true))
        # # Concatenate y_model
        y_model = y_model.detach().cpu()
        y_model_all = torch.cat((y_model_all, y_model))
    
    # From tensor to numpy
    y_model_all = y_model_all.float().numpy()
    y_true_all = y_true_all.int().numpy()
    # Add the auc to the output dict
    output["auc"] = compute_auc(y_true_all.flatten(), y_model_all.flatten())["auc"]

    return output


def compute_auc(y_true, y_model):
    """
    Computes area under the ROC curve
    auc returned should be float
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_model: model outputs (np.array of float32 in [0, 1])

    Note: if you set y_model as the output of solution.Basset, 
    you need to transform it first!
    """
    output = {'auc': 0.}

    # WRITE CODE HERE
    # Check if y_model is a tensor and transform it into numpy
    if torch.is_tensor(y_model):
      y_model = y_model.numpy()

    dic = {'fpr_list': [], 'tpr_list': []}
    K = np.arange(0.0, 1.0, 0.05)
    # Calculate dumb predictions for each K
    for k in K :
      predictions = np.where(y_model >= k, 1, 0)
      dic_k = compute_fpr_tpr(y_true, predictions)
      dic['fpr_list'].append(dic_k['fpr'])
      dic['tpr_list'].append(dic_k['tpr'])
    # compute AUC using the average of left and right reimann sums
    fpr = dic['fpr_list']
    tpr = dic['tpr_list']
    # calculate the right reimann sum
    right_sum = 0
    for i in range(0,len(fpr)-1) :
      right_sum += tpr[i] * (fpr[i+1] - fpr[i])
    # Calculate the left reimann sum
    left_sum = 0
    for i in range(1,len(fpr)) :
      left_sum += tpr[i] * (fpr[i] - fpr[i-1])
    # Add the auc to the output dict
    output['auc'] = np.abs((left_sum + right_sum) / 2)

    return output


def get_critereon():
    """
    Picks the appropriate loss function for our task
    criterion should be subclass of torch.nn
    """

    # WRITE CODE HERE
    critereon = nn.BCEWithLogitsLoss()

    return critereon


def train_loop(model, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to display losses and/or scores within the loop, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!

    Note: you don’t need to compute the score after each training iteration.
    If you do this, your training loop will be really slow!
    You should instead compute it every 50 or so iterations and aggregate ...
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    train_loss = 0.0
    train_auc = 0.0
    model = model.to(device)
    iteration = 0
    batches = len(train_dataloader.dataset)
    auc_partial = []
    
    for i, data in enumerate(train_dataloader):
        sequence, target = data['sequence'].to(device), data['target'].to(device)
        # forward
        predictions = model(sequence)
        # Compute the loss and its gradients
        loss = criterion(predictions, target)
        train_loss += loss.item()
        # zero the parameter gradients
        optimizer.zero_grad()
        # backpropagate
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Re-apply the sigmoid function
        predictions = torch.sigmoid(predictions)

        # # compute auc = score
        if i % 50 == 0:
            iteration += 1
            # from tensor to numpy
            y_model = predictions.detach().float().cpu().numpy()
            y_true = target.int().cpu().numpy()
            # calculate and print auc after 50 iterations
            auc_partial.append(compute_auc(y_true, y_model)['auc'])
            print('For {} batches, score={:.3f}'.format(i+50, np.mean(auc_partial)))
            train_auc += np.mean(auc_partial)
    
    output['total_loss'] = train_loss / batches
    output['total_score'] = train_auc / iteration

    return output['total_score'], output['total_loss']


def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the validation set
    Args:
        :param model: solution.Basset()
        :param valid_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to display losses and/or scores within the loop, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    val_loss = 0.0
    val_auc = 0.0
    model = model.to(device)
    model.eval()
    batches = len(valid_dataloader.dataset)
    iteration = 0  

    with torch.set_grad_enabled(False) :
      for i, data in enumerate(valid_dataloader):
          sequence, target = data['sequence'].to(device), data['target'].to(device)
          # forward
          predictions = model(sequence)
          # Compute the loss and its gradients
          loss = criterion(predictions, target)
          val_loss += loss.item()
          # Re-apply sigmoid
          predictions = torch.sigmoid(predictions)
          # compute auc 
          if i % 50 == 0:
              iteration += 1
              # from tensor to numpy
              y_model = predictions.detach().float().cpu().numpy()
              y_true = target.int().cpu().numpy()
              # calculate auc after 50 iteration
              val_auc += compute_auc(y_true, y_model)['auc']
          
      output['total_score'] = val_auc / iteration
      output['total_loss'] = val_loss / batches

    return output['total_score'], output['total_loss']
