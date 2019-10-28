import torch
import matplotlib.pyplot as plt
import tqdm
import utils
import dataloaders
import numpy as np
import torchvision
import os
from trainer import Trainer
torch.random.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.deterministic = True

'''
### Model Definition
'''

class LeNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        ### START YOUR CODE HERE ### (You can change anything inside this block)
        num_hidden_nodes = 64
        num_classes = 10

        self.classifier = torch.nn.Sequential(
            # First convolutional layer.
            torch.nn.Conv2d(1, 32, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),

            # Second convolutional layer.
            torch.nn.Conv2d(32, 64, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),

            # Third convolutional layer.
            torch.nn.Conv2d(64, 128, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),

            # Flatten inputs before feeding into fully-connected network.
            torch.nn.Flatten(),            

            # Fully-connected network with one hidden layer and an output layer.
            torch.nn.Linear(2048, num_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_nodes, num_classes),
        )
        ### END YOUR CODE HERE ### 

    def forward(self, x):
        ### START YOUR CODE HERE ### (You can change anything inside this block) 
        return self.classifier(x)
        ### END YOUR CODE HERE ### 


'''
### Hyperparameters & Loss function
'''

# Hyperparameters
batch_size = 64
learning_rate = 0.0192
num_epochs = 4


# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

'''
### Train model
'''



image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.25])
])
dataloader_train, dataloader_val = dataloaders.load_dataset(batch_size, image_transform)

# Model definition
model = LeNet()
# Transfer model to GPU memory (if possible)
model = utils.to_cuda(model)

# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate)
trainer = Trainer(
  model=model,
  dataloader_train=dataloader_train,
  dataloader_val=dataloader_val,
  batch_size=batch_size,
  loss_function=loss_function,
  optimizer=optimizer
)
train_loss_dict, val_loss_dict = trainer.train(num_epochs)

'''
### Train Model
'''

utils.plot_loss(train_loss_dict, label="Train Loss")
utils.plot_loss(val_loss_dict, label="Test Loss")
# Limit the y-axis of the plot (The range should not be increased!)
plt.ylim([0, .4])
plt.legend()
plt.xlabel("Global Training Step")
plt.ylabel("Cross Entropy Loss")
os.makedirs("image_processed", exist_ok=True)
plt.savefig(os.path.join("image_processed", "task2.png"))

plt.show()

torch.save(model.state_dict(), "saved_model.torch")



final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_val, model, loss_function)
print(f"Final Validation loss: {final_loss}. Final Validation accuracy: {final_acc}")

