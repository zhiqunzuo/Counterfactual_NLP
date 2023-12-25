import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from data_loading import Augdataset, PretrainDataset
from fair_model import Model, SimpleModel, PretrainModel
from train import train, test, pre_train, pre_test

def create_datasets(max_length: int):
    train_set = PretrainDataset(split="train", max_length=max_length)
    valid_set = PretrainDataset(split="valid", max_length=max_length)
    test_set = PretrainDataset(split="test", max_length=max_length)
    return train_set, valid_set, test_set

def count_tokens(num_tokens: list, dataset) -> list:
    for data_point in dataset:
        mask = data_point[1]
        c_mask = data_point[3]
        num_tokens.append(torch.sum(mask).item())
        num_tokens.append(torch.sum(c_mask).item())
    return num_tokens

def get_max_length() -> int:
    max_length: int = 10000
    length: int = 1000
    while max_length >= length:
        length = length * 2
        train_set, valid_set, test_set = create_datasets(length)
        num_tokens = []
        num_tokens = count_tokens(num_tokens, train_set)
        num_tokens = count_tokens(num_tokens, valid_set)
        num_tokens = count_tokens(num_tokens, test_set)
        max_length = max(num_tokens)
    return max_length + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="experiment_1")
    parser.add_argument("--cuda", type=str, default="7")
    parser.add_argument("--warm_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=500)
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    os.makedirs("results/{}".format(args.name), exist_ok=True)
    
    with open("results/{}/log.txt".format(args.name), "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # if max num tokens unknown
    #max_length: int = get_max_length()
    max_length: int = 135
    batch_size: int = args.batch_size
    
    train_set, valid_set, test_set = create_datasets(max_length)
    print("len train set = {}".format(len(train_set)))
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    #model = Model()
    model = PretrainModel()
    if torch.cuda.is_available():
        model.cuda()
    
    warm_epochs: int = args.warm_epochs
    max_epochs: int = args.max_epochs
    loss_fn = nn.CrossEntropyLoss().cuda()
    lr = 1e-4
    optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-6)
    args = {"batch_size": batch_size, "warm_epochs": warm_epochs, "max_epochs": max_epochs, 
            "loss_fn": loss_fn, "optimizer": optimizer}
    pre_train([train_loader, valid_loader], model, args)
    
    #model = Model()
    model = PretrainModel()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load("results/{}/model_latest.pt".format(args.name))["state_dict"])
    loss, accuracy = pre_test(test_loader, model, args)
    print("*" * 50)
    print("Test Loss: {}\tAccuracy: {}".format(loss, accuracy))

if __name__ == "__main__":
    main()