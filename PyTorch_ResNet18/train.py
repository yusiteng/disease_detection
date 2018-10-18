from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image
import time
import os
import json


# to prepare the dataset

class AgriDataset(Dataset):

    def __init__(self, json_file, root_dir, transform=None):
        self.root_dir = root_dir
        with open(json_file) as f:
            self.dict = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dict[idx]['image_id'])
        image = Image.open(img_name)
        category = self.dict[idx]['disease_class']
        sample = {'img_name': self.dict[idx]['image_id'], 'image': image, 'category': category}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Scale(256),
        #transforms.CenterCrop(224),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

trainset = AgriDataset(json_file='C:/Users/styu/Workspace/ai_challenger_pdr2018_trainingset_20180905/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json',
                       root_dir='C:/Users/styu/Workspace/ai_challenger_pdr2018_trainingset_20180905/AgriculturalDisease_trainingset/images',
                       transform=data_transforms['train']
                       )
valset = AgriDataset(json_file='C:/Users/styu/Workspace/ai_challenger_pdr2018_validationset_20180905/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json',
                     root_dir='C:/Users/styu/Workspace/ai_challenger_pdr2018_validationset_20180905/AgriculturalDisease_validationset/images',
                     transform=data_transforms['val']
                     )

########################################################################
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data['image'], data['category']

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                #print(torch.sum(preds == labels.data))
            epoch_loss = running_loss / dataset_sizes[phase]
            #print(running_corrects)
            epoch_acc = running_corrects.cpu().numpy() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    image_datasets = {'train': trainset, 'val': valset}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=64,
                                                 shuffle=True,
                                                 num_workers=4) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # use gpu or not
    use_gpu = torch.cuda.is_available()

    # get model and replace the original fc layer with your fc layer
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 61)

    if use_gpu:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler)






