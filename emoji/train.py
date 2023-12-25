import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train(loaders, model, args):
    train_loader, valid_loader = loaders
    warm_epochs: int = args.warm_epochs
    max_epochs: int = args.max_epochs
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    weights = torch.tensor([0.7, 0.3]).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weights).cuda()
    
    #for param in model.encoder.parameters():
    #    param.requires_grad = False
    
    losses = []
    valid_losses = []
    valid_accs = []
    
    stops = 0
    best_loss = 10000
    for epoch in range(max_epochs):
        model.train()
        #if epoch >= warm_epochs:
            #for param in model.encoder.parameters():
            #   param.requires_grad = True
            #layer_idx = -1 * (epoch - warm_epochs + 1)
            #layer_idx = -1
            #if layer_idx >= -len(model.encoder.encoder.layer):
            #    for param in model.encoder.encoder.layer[layer_idx].parameters():
            #        param.requires_grad = True
        
        running_loss: float = 0.0
        for i, (tokens, masks, c_tokens, c_masks, labels, sas) in enumerate(train_loader):
            tokens = tokens.squeeze(1).cuda()
            masks = masks.squeeze(1).cuda()
            c_tokens = c_tokens.squeeze(1).cuda()
            c_masks = c_masks.squeeze(1).cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(tokens, masks, c_tokens, c_masks)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch: {}\tLoss:{}".format(epoch, running_loss / len(train_loader)))
        
        valid_loss, valid_acc, valid_fair, valid_unfair = test(valid_loader, model, args)
        print("Validation Loss: {}\tAccuracy: {}\tEqual Opportunity: {}\tUnfairness : {}".format(valid_loss, 
                valid_acc, valid_fair, valid_unfair))
        
        scheduler.step()
        
        losses.append(running_loss / len(train_loader))
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        epochs = range(len(losses))
        plt.plot(epochs, losses, marker="o", color="blue", label="train")
        plt.plot(epochs, valid_losses, marker="s", color="orange", label="validation")
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("results/{}/loss.png".format(args.name))
        plt.close()
        
        plt.plot(epochs, valid_accs, marker="s", color="purple")
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig("results/{}/validation accuracy.png".format(args.name))
        plt.close()
        
        if epoch > warm_epochs:
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({"state_dict": model.module.state_dict()}, "results/{}/model_latest.pt".format(args.name))
                stops = 0
            else:
                stops += 1
            if stops > 10:
                break

def pre_train(loaders, model, args):
    train_loader, valid_loader = loaders
    warm_epochs: int = args.warm_epochs
    max_epochs: int = args.max_epochs
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    weights = torch.tensor([0.7, 0.3]).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weights).cuda()
    
    #for param in model.encoder.parameters():
    #    param.requires_grad = False
    
    losses = []
    valid_losses = []
    valid_accs = []
    
    stops = 0
    best_loss = 10000
    for epoch in range(max_epochs):
        model.train()
        #if epoch >= warm_epochs:
            #for param in model.encoder.parameters():
            #   param.requires_grad = True
            #layer_idx = -1 * (epoch - warm_epochs + 1)
            #layer_idx = -1
            #if layer_idx >= -len(model.encoder.encoder.layer):
            #    for param in model.encoder.encoder.layer[layer_idx].parameters():
            #        param.requires_grad = True
        
        running_loss: float = 0.0
        for i, (tokens, masks, labels) in enumerate(train_loader):
            tokens = tokens.squeeze(1).cuda()
            masks = masks.squeeze(1).cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(tokens, masks)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch: {}\tLoss:{}".format(epoch, running_loss / len(train_loader)))
        
        valid_loss, valid_acc = test(valid_loader, model, args)
        print("Validation Loss: {}\tAccuracy: {}".format(valid_loss, valid_acc))
        
        scheduler.step()
        
        losses.append(running_loss / len(train_loader))
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        epochs = range(len(losses))
        plt.plot(epochs, losses, marker="o", color="blue", label="train")
        plt.plot(epochs, valid_losses, marker="s", color="orange", label="validation")
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("results/{}/loss.png".format(args.name))
        plt.close()
        
        plt.plot(epochs, valid_accs, marker="s", color="purple")
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig("results/{}/validation accuracy.png".format(args.name))
        plt.close()
        
        if epoch > warm_epochs:
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({"state_dict": model.module.state_dict()}, "results/{}/model_latest.pt".format(args.name))
                stops = 0
            else:
                stops += 1
            if stops > 10:
                break
            
        
def get_accuracy(predictions, labels):
    accuracy = (predictions == labels).float().mean()
    return accuracy

def get_fairness(predictions, labels, sas):
    unique_groups = torch.unique(sas)
    equality_rates = {}
    for group in unique_groups:
        group_mask = sas == group
        true_positives = ((predictions == 1) & (labels == 1))
        opportunity_rate = true_positives[group_mask].float().mean()
        equality_rates[group.item()] = opportunity_rate.item()
    return equality_rates

def get_unfairness(equality_rates: list):
    unfairness = abs(equality_rates[0] - equality_rates[1])
    return unfairness

def test(loader, model, args):
    weights = torch.tensor([0.7, 0.3]).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weights).cuda()
    batch_size = args.batch_size
    
    model.eval()
    running_loss: float = 0
    predictions = torch.zeros(len(loader.dataset))
    total_labels = torch.zeros(len(loader.dataset))
    total_sas = torch.zeros(len(loader.dataset))
    for i, (tokens, masks, c_tokens, c_masks, labels, sas) in enumerate(loader):
        tokens = tokens.squeeze(1).cuda()
        masks = masks.squeeze(1).cuda()
        c_tokens = c_tokens.squeeze(1).cuda()
        c_masks = c_masks.squeeze(1).cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(tokens, masks, c_tokens, c_masks)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            predictions[i * batch_size : i * batch_size + prediction.size()[0]] = prediction
            total_labels[i * batch_size : i * batch_size + prediction.size()[0]] = labels
            total_sas[i * batch_size : i * batch_size + prediction.size()[0]] = sas
    accuracy = get_accuracy(predictions, total_labels)
    fairness = get_fairness(predictions, total_labels, total_sas)
    unfairness = get_unfairness(fairness)
    loss = running_loss / len(loader)
    return loss, accuracy, fairness, unfairness

def pre_test(loader, model, args):
    weights = torch.tensor([0.7, 0.3]).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weights).cuda()
    batch_size = args.batch_size
    
    model.eval()
    running_loss: float = 0
    predictions = torch.zeros(len(loader.dataset))
    total_labels = torch.zeros(len(loader.dataset))
    for i, (tokens, masks, labels) in enumerate(loader):
        tokens = tokens.squeeze(1).cuda()
        masks = masks.squeeze(1).cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(tokens, masks)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            predictions[i * batch_size : i * batch_size + prediction.size()[0]] = prediction
            total_labels[i * batch_size : i * batch_size + prediction.size()[0]] = labels
    accuracy = get_accuracy(predictions, total_labels)
    loss = running_loss / len(loader)
    return loss, accuracy