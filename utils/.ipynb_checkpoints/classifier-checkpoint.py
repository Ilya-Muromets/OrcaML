import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
from utils.architectures.inceptionv4 import InceptionV4
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, SmoothL1Loss
from time import gmtime, strftime

class AutoClassMRF(object):
    def __init__(self, batchsize=128, num_epochs=10, num_workers=1, num_classes=2, model=None, model_name=""):
        self.batchsize = batchsize
        self.num_epoch = num_epochs
        self.model = model
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.model_name = model_name

    def fit(self, dataset, test_dataset):
        device = torch.device("cuda")
        curr_date = strftime("%d-%H:%M:%S", gmtime())
        blue = lambda x:'\033[94m' + x + '\033[0m' 

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize,
                                                shuffle=True, num_workers=int(self.num_workers))

        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchsize,
                                                shuffle=True, num_workers=int(self.num_workers))

        print("size of train: ", len(dataset))
        print("size of test: ", len(test_dataset))

        classifier = InceptionV4(self.num_classes)
        classifier.to(device)

        # possibly load model for fine-tuning
        if self.model is not None:
            classifier.load_state_dict(torch.load(self.model))

        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        test_acc = []
        train_loss = []
        val_loss = []

        start = time.time()

        for epoch in range(self.num_epoch):
            for i, data in enumerate(dataloader, 0):
                spectrograms, labels = data[0].type(torch.FloatTensor), data[1].type(torch.LongTensor)
                spectrograms.unsqueeze_(1)
                spectrograms, labels = Variable(spectrograms), Variable(labels)
                spectrograms, labels = spectrograms.to(device), labels.to(device)

                optimizer.zero_grad()
                classifier = classifier.train()
                pred = classifier(spectrograms).view(spectrograms.size()[0],-1)

                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                
                if i % (len(dataloader)/40) == 0:
                    train_loss.append([i + epoch*len(dataloader), loss.item()])
                    print("Loss: ", loss.item())
    
                if i % (len(dataloader)//5) == 0:

                    j, data = next(enumerate(testdataloader, 0))
                    spectrograms, labels = data[0].type(torch.FloatTensor), data[1].type(torch.LongTensor)
                    spectrograms.unsqueeze_(1)
                    spectrograms, labels = Variable(spectrograms), Variable(labels)
                    spectrograms, labels = spectrograms.to(device), labels.to(device)

                    optimizer.zero_grad()
                    classifier = classifier.eval()
                    pred = classifier(spectrograms).view(spectrograms.size()[0],-1)

                    loss = criterion(pred, labels)
                    val_loss.append([i + epoch*len(dataloader), loss.item()])

                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(labels.data).cpu().sum()
                    print(pred_choice[0:10])
                    print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, len(dataloader), blue('test'), loss.item(), correct.item()/float(self.batchsize)))
                    print("Time elapsed: ", (time.time() - start)/60, " minutes")

            if self.model_name is None:
                torch.save(classifier.state_dict(),"models/model" + curr_date + ".mdl")
                np.save("outputs/train_loss" + curr_date, np.array(train_loss))
                np.save("outputs/val_loss" + curr_date, np.array(val_loss))


            else:
                torch.save(classifier.state_dict(),"models/" + self.model_name + ".mdl")
                np.save("outputs/train_loss" + self.model_name, np.array(train_loss))
                np.save("outputs/val_loss" + self.model_name, np.array(val_loss))

