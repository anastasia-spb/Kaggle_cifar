from __future__ import division
from __future__ import print_function

import copy
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models


class InceptionV3ModelWrapper:
    """
    This class loads pretrained Inception v3 model and
    retrain whole model (finetuning)
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """

    def __init__(self, num_classes, device):
        self.device = device
        self.num_classes = num_classes
        # Initialize learning parameters
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 50
        self.batch_size = 32
        self.__initialize_inception_v3_model()
        self.__create_optimizer()

    def __initialize_inception_v3_model(self):
        # We only update the reshaped layer params
        self.model = models.inception_v3(pretrained=True)
        # Handle the auxilary net
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
        # Handle the primary net
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.input_size = 299  # Inception v3 expects image of size (299, 299)
        self.model = self.model.to(self.device)

    def get_expected_img_size(self):
        return self.input_size

    def __create_optimizer(self):
        """
        Gather the parameters to be optimized/updated.
        In feature extract method, we will only update the parameters
        that we have just initialized.
        """
        params_to_update = self.model.parameters()
        self.optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    def train(self, train_dataset, validation_dataset, save_model=True):
        dataloaders = {"train": DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                       "val": DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)}

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for labels, _, inputs in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer_ft.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer_ft.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        if save_model:
            torch.save(self.model.state_dict(), './model_from_last_train.pth')

        return val_acc_history

    def predict(self, torch_dataset, model_from_file=''):
        if model_from_file:
            self.model.load_state_dict(torch.load(model_from_file))

        submission_dl = DataLoader(torch_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        submission_results = []
        img_names_column = []
        for batch_idx, (_, img_names, data) in enumerate(submission_dl):
            ## Forward Pass
            #print(torch.cuda.memory_summary())
            scores, _ = self.model(data.cuda())
            #print(torch.cuda.memory_summary())
            scores = scores.detach()
            #print(torch.cuda.memory_summary())
            softmax = torch.exp(scores).cpu()
            prob = list(softmax.detach().numpy())
            predictions = np.argmax(prob, axis=1)
            # Store results
            submission_results.append(predictions)
            img_names_column.append(img_names)
            #print("Batch idx: ", batch_idx)
            gc.collect()

        return submission_results, img_names_column
