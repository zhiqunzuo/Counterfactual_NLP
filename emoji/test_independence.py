import argparse
import os
import torch
import torch.utils.data as data

from data_loading import Augdataset
from fair_model import TestModel, Discriminator

from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, f1_score

def create_datasets(max_length: int):
    train_set = Augdataset(split="train", max_length=max_length)
    valid_set = Augdataset(split="valid", max_length=max_length)
    test_set = Augdataset(split="test", max_length=max_length)
    return train_set, valid_set, test_set

def feature_extraction(extractor, loader, batch_size):
    extractor.eval()
    feature_size = extractor.encoder.config.hidden_size
    features = torch.FloatTensor(len(loader.dataset), feature_size).cuda()
    f_features = torch.FloatTensor(len(loader.dataset), feature_size).cuda()
    sas = torch.FloatTensor(len(loader.dataset), 1).cuda()
    for i, (tokens, masks, c_tokens, c_mask, label, sa) in enumerate(loader):
        tokens = tokens.cuda()
        masks = masks.cuda()
        c_tokens = c_tokens.cuda()
        c_mask = c_mask.cuda()
        with torch.no_grad():
            rep, f_rep = extractor(tokens, masks, c_tokens, c_mask)
        features[i * batch_size : i * batch_size + tokens.size()[0]] = rep
        f_features[i * batch_size : i * batch_size + tokens.size()[0]] = f_rep
        sas[i * batch_size : i * batch_size + tokens.size()[0]] = sa.unsqueeze(1)
    return features, f_features, sas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--cuda", type=str, default="7")
    parser.add_argument("--batch_size", type=int, default=512)
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.makedirs("independence_results/{}".format(args.name), exist_ok=True)
    
    with open("independence_results/{}/logs.txt".format(args.name), "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    max_length: int = 135
    batch_size: int = args.batch_size
    
    train_set, valid_set, test_set = create_datasets(max_length)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    extractor = TestModel().cuda()
    
    features, f_features, sas = feature_extraction(extractor, train_loader, batch_size)
    torch.save(features, "train_features.pt")
    torch.save(f_features, "train_f_features.pt")
    torch.save(sas, "train_sas.pt")
    features, f_features, sas = feature_extraction(extractor, valid_loader, batch_size)
    torch.save(features, "valid_features.pt")
    torch.save(f_features, "valid_f_features.pt")
    torch.save(sas, "valid_sas.pt")
    features, f_features, sas = feature_extraction(extractor, test_loader, batch_size)
    torch.save(features, "test_features.pt")
    torch.save(f_features, "test_f_features.pt")
    torch.save(sas, "test_sas.pt")
    
    train_features = torch.load("train_features.pt").cpu().numpy()
    train_labels = torch.load("train_sas.pt").cpu().numpy()
    test_features = torch.load("test_features.pt").cpu().numpy()
    test_labels = torch.load("test_sas.pt").cpu().numpy()
    
    clf = SVC(class_weight="balanced")
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features)
    accuracy = f1_score(test_labels, y_pred)
    print(accuracy)
    
    train_f_features = torch.load("train_f_features.pt").cpu().numpy()
    train_f_labels = torch.load("train_sas.pt").cpu().numpy()
    test_f_features = torch.load("test_f_features.pt").cpu().numpy()
    test_f_labels = torch.load("test_sas.pt").cpu().numpy()
    
    clf = SVC(class_weight="balanced")
    clf.fit(train_f_features, train_f_labels)
    y_pred = clf.predict(test_f_features)
    accuracy = f1_score(test_f_labels, y_pred)
    print(accuracy)

if __name__ == "__main__":
    main()