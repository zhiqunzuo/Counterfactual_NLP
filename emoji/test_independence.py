import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.utils.data as data

from data_loading import Augdataset, Multidataset
from fair_model import TestModel, Discriminator, MultiTestModel

from sklearn.svm import SVC 
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def create_datasets(max_length: int):
    train_set = Augdataset(split="train", max_length=max_length)
    valid_set = Augdataset(split="valid", max_length=max_length)
    test_set = Augdataset(split="test", max_length=max_length)
    #train_set = Multidataset(split="train", max_length=max_length)
    #valid_set = Multidataset(split="valid", max_length=max_length)
    #test_set = Multidataset(split="test", max_length=max_length)
    return train_set, valid_set, test_set

def feature_extraction(extractor, loader, batch_size):
    extractor.eval()
    feature_size = extractor.encoder.config.hidden_size
    features = torch.FloatTensor(len(loader.dataset), feature_size).cuda()
    f_features = torch.FloatTensor(len(loader.dataset), feature_size).cuda()
    labels = torch.FloatTensor(len(loader.dataset), 1).cuda()
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
        labels[i * batch_size : i * batch_size + tokens.size()[0]] = label.unsqueeze(1)
        sas[i * batch_size : i * batch_size + tokens.size()[0]] = sa.unsqueeze(1)
    return features, f_features, labels, sas

def multi_feature_extraction(extractor, loader, batch_size):
    extractor.eval()
    feature_size = extractor.encoder.config.hidden_size
    features = torch.FloatTensor(len(loader.dataset), feature_size).cuda()
    f_features = torch.FloatTensor(len(loader.dataset), feature_size).cuda()
    labels = torch.FloatTensor(len(loader.dataset), 1).cuda()
    sas = torch.FloatTensor(len(loader.dataset), 1).cuda()
    for i, (tokens, masks, c1_tokens, c1_mask, c2_tokens, c2_mask, 
            c3_tokens, c3_mask, c4_tokens, c4_mask, c5_tokens, c5_mask, label, sa) in enumerate(loader):
        tokens = tokens.cuda()
        masks = masks.cuda()
        c1_tokens = c1_tokens.cuda()
        c1_mask = c1_mask.cuda()
        c2_tokens = c2_tokens.cuda()
        c2_mask = c2_mask.cuda()
        c3_tokens = c3_tokens.cuda()
        c3_mask = c3_mask.cuda()
        c4_tokens = c4_tokens.cuda()
        c4_mask = c4_mask.cuda()
        c5_tokens = c5_tokens.cuda()
        c5_mask = c5_mask.cuda()
        with torch.no_grad():
            rep, f_rep = extractor(tokens, masks, c1_tokens, c1_mask, c2_tokens, c2_mask, c3_tokens,
                                   c3_mask, c4_tokens, c4_mask, c5_tokens, c5_mask)
        features[i * batch_size : i * batch_size + tokens.size()[0]] = rep
        f_features[i * batch_size : i * batch_size + tokens.size()[0]] = f_rep
        labels[i * batch_size : i * batch_size + tokens.size()[0]] = label.unsqueeze(1)
        sas[i * batch_size : i * batch_size + tokens.size()[0]] = sa.unsqueeze(1)
    return features, f_features, labels, sas

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
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    extractor = TestModel().cuda()
    
    """features, f_features, labels, sas = feature_extraction(extractor, train_loader, batch_size)
    torch.save(features, "train_features.pt")
    torch.save(f_features, "train_f_features.pt")
    torch.save(labels, "train_labels.pt")
    torch.save(sas, "train_sas.pt")
    features, f_features, labels, sas = feature_extraction(extractor, valid_loader, batch_size)
    torch.save(features, "valid_features.pt")
    torch.save(f_features, "valid_f_features.pt")
    torch.save(labels, "valid_labels.pt")
    torch.save(sas, "valid_sas.pt")
    features, f_features, labels, sas = feature_extraction(extractor, test_loader, batch_size)
    torch.save(features, "test_features.pt")
    torch.save(f_features, "test_f_features.pt")
    torch.save(labels, "test_labels.pt")
    torch.save(sas, "test_sas.pt")"""
    
    train_factual = torch.load("train_features.pt").cpu().numpy()
    train_counter = torch.load("train_f_features.pt").cpu().numpy()
    train_features = np.concatenate((train_factual, train_counter), axis=0)
    factual_label = np.zeros(train_factual.shape[0])
    counter_label = np.ones(train_counter.shape[0])
    train_labels = np.concatenate((factual_label, counter_label), axis=0)
    
    test_factual = torch.load("test_features.pt").cpu().numpy()
    test_counter = torch.load("test_f_features.pt").cpu().numpy()
    test_features = np.concatenate((test_factual, test_counter), axis=0)
    factual_label = np.zeros(test_factual.shape[0])
    counter_label = np.ones(test_counter.shape[0])
    test_labels = np.concatenate((factual_label, counter_label), axis=0)
    
    clf = SVC(class_weight="balanced")
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features)
    cm = confusion_matrix(test_labels, y_pred)
    print("confusion matrix:\n{}".format(cm))
    acc = accuracy_score(test_labels, y_pred)
    print("Accuracy = {}".format(acc))
    
    #train_features = torch.load("train_features.pt").cpu().numpy()
    #train_labels = torch.load("train_labels.pt").cpu().numpy()
    #train_sas = torch.load("train_sas.pt").cpu().numpy()
    #test_features = torch.load("test_features.pt").cpu().numpy()
    #test_labels = np.squeeze(torch.load("test_labels.pt").cpu().numpy())
    #test_sas = np.squeeze(torch.load("test_sas.pt").cpu().numpy())
    
    #train_f_features = torch.load("train_f_features.pt").cpu().numpy()
    #train_f_labels = torch.load("train_labels.pt").cpu().numpy()
    #train_f_sas = torch.load("train_sas.pt").cpu().numpy()
    #test_f_features = torch.load("test_f_features.pt").cpu().numpy()
    #test_f_labels = np.squeeze(torch.load("test_labels.pt").cpu().numpy())
    #test_f_sas = np.squeeze(torch.load("test_sas.pt").cpu().numpy())
    
    """train_sas = train_sas.ravel()
    test_sas = test_sas.ravel()
    train_f_sas = train_f_sas.ravel()
    test_f_sas = test_f_sas.ravel()
    
    clf = SVC(class_weight="balanced")
    clf.fit(train_features, train_sas)
    s_pred = clf.predict(test_features)
    cm = confusion_matrix(test_sas, s_pred)
    print("Factual Confusion Matrix:\n{}".format(cm))
    acc = accuracy_score(test_sas, s_pred)
    print("Factual Accuracy = {}".format(acc))
    
    clf = SVC(class_weight="balanced")
    clf.fit(train_f_features, train_f_sas)
    s_pred = clf.predict(test_f_features)
    cm = confusion_matrix(test_f_sas, s_pred)
    print("Counter Confusion Matrix:\n{}".format(cm))
    acc = accuracy_score(test_f_sas, s_pred)
    print("Counter Accuracy = {}".format(acc))"""
    
    #tsne = TSNE(n_components=2, random_state=42)
    #transformed_features = tsne.fit_transform(train_features)
    #plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=train_sas)
    #plt.colorbar()
    #plt.savefig("tsne_features.png")
    #plt.close()
    #tsne = TSNE(n_components=2, random_state=42)
    #transformed_f_features = tsne.fit_transform(train_f_features)
    #plt.scatter(transformed_f_features[:, 0], transformed_f_features[:, 1], c=train_f_sas)
    #plt.colorbar()
    #plt.savefig("tsne_f_features.png")
    #plt.close()
    
    """train_labels = train_labels.ravel()
    train_f_labels = train_f_labels.ravel()
    train_sas = train_sas.ravel()
    train_f_sas = train_f_sas.ravel()
    
    test_labels = test_labels.ravel()
    test_f_labels = test_f_labels.ravel()
    test_sas = test_sas.ravel()
    test_f_sas = test_f_sas.ravel()
    
    clf = SVC(class_weight="balanced")
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features)
    group_1_prediction = y_pred[test_sas == 0]
    group_2_prediction = y_pred[test_sas == 1]
    print("In group 1:")
    print("{}% predictions are 0".format(sum(group_1_prediction) / group_1_prediction.size))
    print("{}% predictions are 1".format(sum(1 - group_1_prediction) / group_1_prediction.size))
    print("In group 2:")
    print("{}% predictions are 0".format(sum(group_2_prediction) / group_2_prediction.size))
    print("{}% predictions are 1".format(sum(1 - group_2_prediction) / group_2_prediction.size))
    
    clf = SVC(class_weight="balanced")
    clf.fit(train_f_features, train_f_labels)
    y_pred = clf.predict(test_f_features)
    group_1_prediction = y_pred[test_f_sas == 0]
    group_2_prediction = y_pred[test_f_sas == 1]
    print("In group 1:")
    print("{}% predictions are 0".format(sum(group_1_prediction) / group_1_prediction.size))
    print("{}% predictions are 1".format(sum(1 - group_1_prediction) / group_1_prediction.size))
    print("In group 2:")
    print("{}% predictions are 0".format(sum(group_2_prediction) / group_2_prediction.size))
    print("{}% predictions are 1".format(sum(1 - group_2_prediction) / group_2_prediction.size))"""
    
    #train_sas = train_sas.ravel()
    #train_f_sas = train_f_sas.ravel()
    #test_sas = test_sas.ravel()
    #test_f_sas = test_f_sas.ravel()
    
    #clf = SVC(class_weight="balanced")
    #clf.fit(train_features, train_sas)
    #s_pred = clf.predict(test_features)
    #cm = confusion_matrix(test_sas, s_pred)
    #print(cm)
    #accuracy = f1_score(test_sas, s_pred)
    #print("accuracy = {}".format(accuracy))
    
    #clf = SVC(class_weight="balanced")
    #clf.fit(train_f_features, train_f_sas)
    #s_pred = clf.predict(test_f_features)
    #cm = confusion_matrix(test_f_sas, s_pred)
    #print(cm)
    #accuracy = f1_score(test_f_sas, s_pred)
    #print("accuracy = {}".format(accuracy))
    
    """clf = SVC(class_weight="balanced")
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features)
    accuracy = f1_score(test_labels, y_pred)
    print(accuracy)
    
    y_pred_group_1 = y_pred[test_sas == 1]
    y_pred_group_2 = y_pred[test_sas == 0]
    print("group 1 ratio = {}".format(np.sum(y_pred_group_1) / y_pred_group_1.size))
    print("group 2 ratio = {}".format(np.sum(y_pred_group_2) / y_pred_group_2.size))
    
    train_f_features = torch.load("train_f_features.pt").cpu().numpy()
    train_f_labels = torch.load("train_sas.pt").cpu().numpy()
    test_f_features = torch.load("test_f_features.pt").cpu().numpy()
    test_f_labels = np.squeeze(torch.load("test_labels.pt").cpu().numpy())
    test_f_sas = np.squeeze(torch.load("test_sas.pt").cpu().numpy())
    
    clf = SVC(class_weight="balanced")
    clf.fit(train_f_features, train_f_labels)
    y_pred = clf.predict(test_f_features)
    accuracy = f1_score(test_f_labels, y_pred)
    print(accuracy)
    
    y_pred_group_1 = y_pred[test_f_sas == 1]
    y_pred_group_2 = y_pred[test_f_sas == 0]
    print("group 1 ratio = {}".format(np.sum(y_pred_group_1) / y_pred_group_1.size))
    print("group 2 ratio = {}".format(np.sum(y_pred_group_2) / y_pred_group_2.size))"""

if __name__ == "__main__":
    main()