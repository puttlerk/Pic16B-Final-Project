import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# save and load model functions
save_model = lambda model, name: torch.jit.script(model).save(name)
load_model = lambda name: torch.jit.load(name)

# creates a list of labels
labels_names = [chr(ord('A') + i) for i in range(26)]
labels_names.extend(["del","nothing","space"])

def evaluate(model, test_loader, loss_criteria, device):
    """
    Tests the model's accuracy

    @params:
        model: pytorch nn.Module, model to evaluate
        test_loader: pytorch DataLoader, data to evaluate on
        loss_criteria: pytorch loss function, loss function

    @returns:
        (accuracy, test_loss) floats representing the percentage accuracy and average loss
    """
    
    # Sets model to eval mode
    model.eval()

    correct = 0
    test_loss = 0
    total_samples = 0
    
    with torch.no_grad():

        for data, label in test_loader:
            # Move data, label to device
            data = data.to(device)
            label = label.to(device)
            
            # Model's prediction
            output = model(data)
            
            # Calculate the number of correct responses in current batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(label==predicted).item()
            test_loss += loss_criteria(output, label).item()
            total_samples += data.size(0)
    
    # Get average loss
    test_loss = test_loss / total_samples
    # Get accuracy
    accuracy = round((100 * correct / len(test_loader.dataset)),2)
    
    return (accuracy, test_loss)


def train(model, train_loader, test_loader, optimizer, loss_criteria, epoch, device = "cpu"):
    """
    Trains the model parameters and returns average loss and percentage accuracy 
    
    @params:
        model - pytorch model to train
        train_loader - dataLoader object w/ training data
        test_loader - dataLoader object w/ testing data
        optimizer - pytorch optimizer object
        loss_criteria - pytorch loss criteria
        epoch - what the current epoch is
        device - device to use for computation
    
    @returns:
        (train_loss, test_loss, train_acc, test_acc) each a float 
    """
    
    # sets the model to training mode
    model.train()
    loss_tracker = 0
    correct = 0
    total_samples = 0
    
    # progress bar
    train_loader_iter = tqdm.tqdm(train_loader, desc = f"Epoch {epoch + 1}")
    
    for data, label in train_loader_iter:

        data = data.to(device)
        label = label.to(device)

        # resets the gradients
        optimizer.zero_grad()
        
        # model's current prediction
        prediction = model(data)
        
        # tracks loss
        loss = loss_criteria(prediction, label)
        loss_tracker += loss.item()
        total_samples += data.size(0)
        
        # calculates gradient
        loss.backward()
        optimizer.step()
        
        # accuracy
        _, predicted = torch.max(prediction.data, 1)
        correct += torch.sum(label == predicted).item()
        
    # get training accuracy and average training loss
    train_acc = round((100 * correct / len(train_loader.dataset)), 2)
    train_loss =  loss_tracker / total_samples
    
    # get test accuracy and average loss
    test_acc, test_loss = evaluate(model, test_loader = test_loader, loss_criteria = loss_criteria, device = device)
        
    return (train_loss, test_loss, train_acc, test_acc)  

def make_plots(title, train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Creates a losses and accuracies plots and saves them to specified file name.

    @params:
        file_name: string, name with which to save plot 
        train_losses: list of floats
        test_losses: list of floats
        train_accuracies: list of floats
        test_accuracies: list of floats
    """

    # make sure all the lists have the same length
    assert(len(train_losses) == len(test_losses))
    assert(len(test_losses) == len(train_accuracies))
    assert(len(train_accuracies) == len(test_accuracies))

    # get range of horizontal dimension (epochs) for plots
    epochs = len(train_losses)

    # create subplots
    fig, ax = plt.subplots(1, 2, figsize = (20, 8))
    
    # plot the losses
    ax[0].plot(np.arange(1, epochs + 1), train_losses, label = "Train Loss")
    ax[0].plot(np.arange(1, epochs + 1), test_losses, label = "Test Loss")
    ax[0].set(title = "Losses over Epochs", ylabel = "Loss", xlabel = "Epoch")
    ax[0].set_xticks(np.arange(1, epochs + 1, 3));
    ax[0].legend(loc = "upper right")

    # plot the accuracies
    ax[1].plot(np.arange(1,epochs + 1), train_accuracies, label = "Train Accuracy")
    ax[1].plot(np.arange(1,epochs + 1), test_accuracies, label = "Test Accuracy")
    ax[1].set(title = "Accuracies over Epochs", ylabel = "Accuracy %", xlabel = "Epoch")
    ax[1].set_xticks(np.arange(1, epochs + 1, 3));
    ax[1].legend(loc = "upper right")

    # title the plot
    plt.title(title)

    # save plot
    plt.savefig(title, format="png")

    # close the figure
    plt.close()


def create_confusion_matrix(title, model, test_loader, device):
    """ 
    Creates a confusion matrix given a model and a dataLoader object
    
    @params:
        model: pytorch nn.Module, model for which the matrix is made
        test_loader: pytorch dataloader, data with which the matrix is made
    """
    
    # sets the model to eval mode
    model.eval()

    labels_list = []
    predicted_list = []
    
    with torch.no_grad():
    
        for data, label in test_loader:
            # fills the labels and predictions
            labels_list.extend(label.numpy())
            # convert to device to calculate
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            # convert back to cpu to do numpy
            predicted_list.extend(predicted.cpu().numpy())
            
    # construct the confusion matrix 
    conf_matrix = confusion_matrix(labels_list, predicted_list)
    
    # creates a sns heatmap with the conf matrix
    plt.figure(figsize=(14, 10))
    sns.heatmap(conf_matrix, annot = True, cmap = 'Blues', fmt = 'g', 
            xticklabels = labels_names, yticklabels = labels_names, vmax = 100)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # saves the figure
    plt.savefig(title, format = "png")

    # close the figure
    plt.close()