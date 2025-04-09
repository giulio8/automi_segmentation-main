import pickle
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
import os
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def save_model_params(fpath, model_params):
    with open(os.path.join(fpath,'modelparams.pkl'), 'wb') as f:
        pickle.dump(model_params, f)


def getLossWeights_None(train_loader, index):
    return {}, 1


def getLossWeights_PerClass(train_loader, index):
    if train_loader.dataset.samples[0][1].dim() > 1:
        weight = torch.zeros(2, 1)
        nclasses = 1
        for img_list in train_loader.dataset.samples:
            cl = int(img_list[1][index])
            weight[cl] += 1
    else:
        weight = torch.zeros(2, len(train_loader.dataset.samples[0][1]))
        for img_list in train_loader.dataset.samples:
            for index, cl in enumerate(img_list[1]):
                weight[int(cl), index] += 1

    weight = weight[0, :] / weight[1, :]
    if int(torch.sum(weight == 0)) > 0:
        raise NameError('Found 0 cases on label!')

    return {'weight': weight.cuda()}, len(weight)


class dataPredictions(object):
    '''
    This class is used to save the predictions of the model during training.
    '''
    def __init__(self, fpath='.'):
        self.predictions = []
        self.fpath = fpath

    def append(self, input, output, target=None, info=None):
        if len(self.predictions) < 100:
            for ind, (i, o, info) in enumerate(zip(input, output, info)):
                i = np.squeeze(i.cpu().detach().numpy())
                o = np.squeeze(o.cpu().detach().numpy())
                if target is None:
                    t = np.zeros(i.shape)
                else:
                    t = np.squeeze(target[ind].cpu().detach().numpy())
                if info is None:
                    rpath = f'{len(self.predictions)}'
                else:
                    rpath = f'{info}'

                self.predictions.append([i, o, t, rpath])

    def write(self, fpath=None, mode='output', nexamples=10, clear=False):
        if fpath is None:
            if self.fpath is None:
                print('Could not write predictions to empty fpath.')
                return
            else:
                fpath = self.fpath

        for ind, (i, o, t, fname) in enumerate(self.predictions):
            fname = fname.split('.')[0]
            i = ((i - np.min(i)) / (np.max(i) - np.min(i))) * 255
            i = i.astype(np.uint8)
            o = o * 255
            o = o.astype(np.uint8)
            t = t * 255
            t = t.astype(np.uint8)
            if mode == 'output':
                # If the number of outputs is 1
                Image.fromarray(i).save(os.path.join(fpath, f'{fname}.png'))
                Image.fromarray(o).save(os.path.join(fpath, f'{fname}_mask.png'))
                Image.fromarray(t).save(os.path.join(fpath, f'{fname}_target.png'))
            elif mode == 'examples':
                if np.max(t) > 0 and t.shape[0] != 3:
                    Image.fromarray(np.concatenate((i, o, t), axis=1)).save(os.path.join(fpath, f'{fname}.png'))
                if ind == nexamples - 1:
                    return

        if clear:
            self.predictions = []



class DataloaderIterator(object):
    def __init__(self, mode, calcLoss, calcAcc, dataloader, n_batches=None, datapredictions=dataPredictions()):
        self.mode = mode
        self.calcLoss = calcLoss
        self.calcAcc = calcAcc
        self.dataloader = dataloader
        if n_batches is None:
            self.nbatches = len(dataloader)
        else:
            self.nbatches = min(n_batches, len(dataloader))
        self.predictions = datapredictions

    def __call__(self, model, optimizer, epoch):
        self.loss = 0
        self.acc = 0
        self.nsamples = 0
        self.predictions.predictions = []
        self.start = timer()
        self.elapsed = 0
        for ii, (data, target, info) in enumerate(self.dataloader):

            # Useful for debbuging, can see the images and the weights and see what is going on
            # for d,t in zip(data, target):
            #     plt.figure()
            #     plt.imshow(d[0].numpy())
            #     plt.figure()
            #     plt.imshow(t[0].numpy())
            #     plt.show()

            # Tensors to gpu
            data = data.cuda()
            target = target.cuda()
            model = model.cuda()


            if self.mode == 'train':
                # Clear gradients
                optimizer.zero_grad()

            # Predicted output
            output = model(data)

            # Get loss:
            loss = self.calcLoss(output, target)

            # Update the parameters
            if self.mode == 'train':
                loss.backward()
                optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            try:
                datasize = data.size(0)
            except:
                datasize = data[0].size(0)
            self.loss += loss.item() * datasize
            self.nsamples += datasize

            # Calculate accuracy
            self.acc += self.calcAcc(output, target)

            if self.mode == 'val':
                self.predictions.append(data, output, target=target, info=info)

            # Track progress
            # print('Epoch')
            print(
                f'\rEpoch {self.mode}: {epoch}\t{100 * (ii + 1) / self.nbatches:.2f}% complete - loss {self.loss / self.nsamples:.4f}. {timer() - self.start:.2f} seconds elapsed in epoch.',
                end='')
            if (ii + 1) == self.nbatches:
                break

        self.loss = self.loss / self.nsamples
        self.acc = self.acc / self.nsamples
        print(f'\nEpoch: {epoch} \t{self.mode} Loss: {self.loss:.4f} \t{self.mode} Acc: {self.acc:.4f}')
        self.elapsed = timer() - self.start


class ClassAcc(object):
    def __call__(self, output, target):
        pred = torch.round(output)
        try:
            target = target.cuda()
        except:
            pass
        correct_tensor = pred.eq(target.data.view_as(pred))
        return torch.mean(correct_tensor.type(torch.FloatTensor)).item() * output.size(0)


class GenericAcc(object):
    def __init__(self, acc_func, indlabels=None, singleOutput=None):
        if indlabels == None:
            self.indlabels = [i for i in range(len(acc_func))]
        else:
            self.indlabels = indlabels
        self.nlabels = len(self.indlabels)

        self.func = [accf() for accf in acc_func]

        if singleOutput == None:
            if len(self.indlabels) == 1 and self.indlabels[0] == 0 and len(self.func) == 1:
                singleOutput = True
            else:
                singleOutput = False
        self.singleOutput = singleOutput

    def __call__(self, output, target):
        if self.singleOutput:
            acc = self.func[0](output, target)
        else:
            acc = 0
            for ind, func in zip(self.indlabels, self.func):
                acc += func(output[ind], target[ind])

        return acc / self.nlabels


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets):

            # Uncomment this to show the input, weights and target of the training iteration
            # if ind == 0:
            #     plt.close()
            #     plt.figure()
            #     plt.title("Input" + imageDataset)
            #     plt.imshow(inputs[index].cpu().detach().numpy())
            #     plt.figure()
            #     plt.title("Targets"+ imageDataset)
            #     plt.imshow(targets[index].cpu().detach().numpy(),vmin=0, vmax=1)
            #     plt.show()

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return loss

    def createBoundingBoxWeights(self, targets):
        emptyMask = torch.ones(targets.size(dim=0), targets.size(dim=1), targets.size(dim=2), targets.size(dim=3))
        weights = emptyMask + targets
        return weights

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice_loss = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return 1 - dice_loss + bce_loss


class GenericLoss(object):
    def __init__(self, train_loader, weight_func=[getLossWeights_None], criterion=[BCELoss],
                 loss_weights=None, index_labels=None, singleOutput=None):
        if index_labels == None:
            index_labels = [i for i in range(len(train_loader.dataset.allMasksPath))]
        elif not isinstance(index_labels, list):
            index_labels = [index_labels]
        self.index_labels = index_labels
        self.nlabels = len(self.index_labels)

        wfuncdict = []
        for ind, wfunc in zip(self.index_labels, weight_func):
            wdict, _ = wfunc(train_loader, ind)
            wfuncdict.append(wdict)

        self.criterion = [crit(**wdict) for crit, wdict in zip(criterion, wfuncdict)]

        if singleOutput == None:
            if len(self.index_labels) == 1 and self.index_labels[0] == 0 and len(self.criterion) == 1:
                singleOutput = True
            else:
                singleOutput = False
        self.singleOutput = singleOutput

        if loss_weights == None:
            self.loss_weights = [1 for _ in self.criterion]
        else:
            self.loss_weights = loss_weights
        self.loss_weights = [lw / max(self.loss_weights) for lw in self.loss_weights]

    def __call__(self, output, target):
        if self.singleOutput:
            for ind, criterion, crit_weight in zip(self.index_labels, self.criterion, self.loss_weights):
                o = output
                t = target

                # Useful for debugging
                # oMax = torch.max(o[0][0])
                # oMin = torch.min(o[0][0])
                # tMax = torch.max(t[0][0])
                # tMin = torch.min(t[0][0])
                # plt.figure()
                # plt.title("Output")
                # plt.imshow(o[0][0].cpu().detach())
                # plt.figure()
                # plt.title("Target")
                # plt.imshow(t[0][0].cpu().detach())
                # plt.show()
                try:
                    t = t.cuda()
                    o = o.cuda()
                except:
                    pass
                loss = crit_weight * criterion(o, t)
        else:
            loss = 0
            for ind, criterion, crit_weight in zip(self.index_labels, self.criterion, self.loss_weights):

                o = output[:, ind, :, :]
                t = target[:, ind, :, :]
                # oMax = torch.max(o)
                # oMin = torch.min(o)
                # tMax = torch.max(t)
                # tMin = torch.min(t)
                # plt.figure()
                # plt.title("Output Index: {}".format(ind))
                # plt.imshow(o[0].cpu().detach())
                # plt.figure()
                # plt.title("Target")
                # plt.imshow(t[0].cpu().detach())
                # plt.show()
                try:
                    t = t.cuda()
                    o = o.cuda()
                except:
                    pass
                loss_ind = crit_weight * criterion(o, t)
                loss += loss_ind / self.nlabels
        return loss


def train(model, train_loader, valid_loader,
          save_file_name, lossCriterion, accCriterion,
          valPredictions=dataPredictions(), learning_rate=1e-4,
          max_epochs_stop=3, n_epochs=20, n_batches=None):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping initialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    num_unfreeze_layer = 1
    history = []
    overall_start = timer()
    if not hasattr(model, 'epochs'):
        model.epochs = 0

    # Main loop
    trainIt = DataloaderIterator('train', lossCriterion, accCriterion, train_loader, n_batches=n_batches)
    valIt = DataloaderIterator('val', lossCriterion, accCriterion, valid_loader, n_batches=n_batches,
                               datapredictions=valPredictions)
    for epoch in range(n_epochs):
        # Training loop
        model.train()  # Set to training mode
        trainIt(model, optimizer, epoch)
        model.epochs += 1

        # Validation loop
        with torch.no_grad():  # Don't need to keep track of gradients
            model.eval()  # Set to evaluation mode
            valIt(model, optimizer, epoch)

            # write history at the end of each epoch!
            history.append([trainIt.loss, valIt.loss, trainIt.acc, valIt.acc])

            # Save the model if validation loss decreases
            if valIt.loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valIt.loss
                valid_best_acc = valIt.acc
                best_epoch = epoch
                # Write predictions for best model
                valIt.predictions.write(mode='examples')
            else:  # Otherwise increment count of epochs with no improvement
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%')
                    total_time = timer() - overall_start
                    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.')
                    break

        # Report back minimum estimated time
        epochs_missing = max_epochs_stop - epochs_no_improve
        estimated_time = epochs_missing * (trainIt.elapsed + valIt.elapsed)
        print(f'Check back in {estimated_time:.2f} seconds.')

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%')
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.')

    return model, history


def save_checkpoint(model, multi_gpu, path):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    model_name = path.split('-')[0]

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the final classifier and the state dictionary
    if 'vgg' in model_name or 'alexnet' in model_name or 'densenet' in model_name:
        # Check to see if model was parallelized
        if multi_gpu:
            checkpoint['classifier'] = model.module.classifier
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['classifier'] = model.classifier
            checkpoint['state_dict'] = model.state_dict()

    else:
        if multi_gpu:
            checkpoint['fc'] = model.module.fc
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['fc'] = model.fc
            checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)
