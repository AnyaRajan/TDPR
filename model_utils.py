import models
from data_util import get_train_data, get_val_and_test

def initialize_model():
    return models.__dict__[conf.model]().to(device)

def get_data_loaders():
    trainloader = get_train_data(conf.dataset)
    valloader, testloader = get_val_and_test(conf.corruption)
    return trainloader, valloader, testloader
