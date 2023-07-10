import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dataset_root = 'datasets/game_state_tracker'
model_root = 'models/'
model_name= 'game_state_tracker.pth'

num_epochs = 200

class GameStateTracker:
    def __init__(self, model_name=model_name):
        self.model = GameStateTrackerModel()
        self.model.load_state_dict(torch.load(model_root + model_name))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def get_state(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_label = torch.argmax(output, dim=1).item()

        match predicted_label:
            case 0:
                return 'playing'
            case 1:
                return 'ready'
            case 2:
                return 'lose'
            case 3:
                return 'lose_waiting'
            case 4:
                return 'loading'
            case 5:
                return 'pause'
        
class GameStateTrackerModel(nn.Module):
    def __init__(self):
        super(GameStateTrackerModel, self).__init__()

        # Define the CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Define the fully connected layers for processing concatenated features
        self.fc = nn.Sequential(
            nn.Linear(50176, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        features = self.cnn(x)
        features = torch.flatten(features, 1)
        output = self.fc(features)
        return output
    

class GameStateTrackerDataset(Dataset):
    def __init__(self, data_path, transform, dataset_size):
        self.data_path = data_path
        self.transform = transform
        self.length = dataset_size
        
        self.image_paths = self.load_image_paths()
        self.additional_values = self.load_additional_values()

    def load_image_paths(self):
        image_paths = []
        for i in range(1, len(self) + 1):
            image_path = os.path.join(self.data_path, f"{str(i).zfill(3)}.png")
            image_paths.append(image_path)
        return image_paths

    def load_additional_values(self):
        additional_values = []
        positions_file = os.path.join(self.data_path, "labels.txt")
        with open(positions_file, 'r') as f:
            for line in f:
                value = line.strip()
                one_hot_value = [int(value), 1 - int(value)]
                additional_values.append(one_hot_value)
        return additional_values

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # Load the image
        image = Image.open(image_path)

        # Convert the image to a Tensor
        image = torchvision.transforms.ToTensor()(image)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Get the additional value associated with the image
        value = self.additional_values[index]
        value = torch.tensor(value[0], dtype=torch.long)

        return image, value
    
def train_model(model_name=model_name, load_model=False):
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = GameStateTrackerDataset(dataset_root + '/train', transform, dataset_size=177)    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Define model
    model = GameStateTrackerModel()
    model.train()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    lowest_average_loss = 9999999999

    for epoch in range(num_epochs):
        
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)

        if epoch % 25 == 0:
            torch.save(model.state_dict(), model_root + f'game_state_tracker_{epoch}.pth')

        if (average_loss < lowest_average_loss):
            lowest_average_loss = average_loss
            torch.save(model.state_dict(), model_root + 'game_state_tracker_lowest_loss.pth')

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')


    # Save the trained model
    torch.save(model.state_dict(), model_root + model_name)
    print(f"Trained model saved at '{model_root + model_name}'")


def eval_model(model_name=model_name):
    
    print(f'Evaluating {model_name}')

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create the dataset and dataloader
    dataset = GameStateTrackerDataset(dataset_root + "/test", transform, dataset_size=83)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the trained model
    model = GameStateTrackerModel()
    model.load_state_dict(torch.load(model_root + model_name))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.eval()

    num_correct = 0
    num_total = 0

    # Evaluate the model on the training images
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Forward pass
            output = model(images)

            # Convert the predicted coordinates to numpy arrays
            predicted_labels = torch.argmax(output, dim=1)
            predicted_label = predicted_labels[0].item()
            real_label = labels.numpy()
            num_total += 1
            if real_label == predicted_label:
                num_correct += 1

            # Print the real and predicted coordinates, as well as the percent difference
            print(f'Real: {real_label}, Predicted: {predicted_label}, Correct: {real_label == predicted_label}')
    print(f'{num_correct} correct out of {num_total} - {100 * num_correct/num_total}%')

if (__name__ == '__main__'):
    # train_model()
    eval_model()