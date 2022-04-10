import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def confusion_matrix(model, inputs, target, path, model_type = None):
    conf_matrix = torch.zeros(10, 10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf_matrix.to(device)
    with torch.no_grad():
        if model_type == "CNN":
            inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1
    total = np.zeros((10, 1))
    for k in range(10):
        total[k] = (target.detach().cpu().numpy() == k).sum()
    conf_matrix = conf_matrix/total
    plt.figure(constrained_layout=True, figsize=(10, 10))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2g')

    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
    ax.yaxis.set_ticklabels(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])

    ## Display the visualization of the Confusion Matrix.
    plt.savefig(path)
        
