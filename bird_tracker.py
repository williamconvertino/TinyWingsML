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

dataset_root = 'datasets/bird_tracker'
model_path= 'models/bird_tracker.pth'

class BirdTracker:
    def __init__(self, model_path=model_path):
        self.model = BirdTrackerModel()
        self.model.load_state_dict(torch.load(model_path))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def get_coords(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_coords = output.cpu().numpy()
        x_coord, y_coord = abs(predicted_coords[0])
        
        return int(x_coord), int(y_coord)
    

class BirdTrackerModel(nn.Module):
    def __init__(self):
        super(BirdTrackerModel, self).__init__()

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
            nn.Linear(128, 2)  # Output 2 values for x and y coordinates
        )

    def forward(self, x):
        features = self.cnn(x)
        features = torch.flatten(features, 1)
        output = self.fc(features)
        return output
    

class BirdTrackerDataset(Dataset):
    def __init__(self, data_path, transform, dataset_size):
        self.data_path = data_path
        self.transform = transform
        self.length = dataset_size
        
        self.image_paths = self.load_image_paths()
        self.additional_values = self.load_additional_values()

    def load_image_paths(self):
        image_paths = []
        for i in range(1, len(self) + 1):
            image_path = os.path.join(self.data_path, f"{i}.png")
            image_paths.append(image_path)
        return image_paths

    def load_additional_values(self):
        additional_values = []
        positions_file = os.path.join(self.data_path, "positions.txt")
        with open(positions_file, 'r') as f:
            for line in f:
                x, y = line.strip().split(',')
                additional_values.append((int(x), int(y)))
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
        coords = self.additional_values[index]
        coords = torch.tensor(coords, dtype=float)

        return image, coords
    
def train_model(model_path=model_path, load_model=False):
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = BirdTrackerDataset(dataset_root + '/train', transform, dataset_size=140)    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Define model
    model = BirdTrackerModel()
    model.train()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    num_epochs = 30
    
    for epoch in range(num_epochs):
        
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')
        
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at '{model_path}'")


def eval_model(model_path=model_path):
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create the dataset and dataloader
    dataset = BirdTrackerDataset(dataset_root + "/test", transform, dataset_size=16) 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the trained model
    model = BirdTrackerModel()
    model.load_state_dict(torch.load(model_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.eval()

    # Initialize lists to store percentage difference values (for analyticals)
    x_percent_diff = []
    y_percent_diff = []

    # Evaluate the model on the training images
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Convert the predicted coordinates to numpy arrays
            predicted_coords = outputs.cpu().numpy()
            real_coords = labels.numpy()

            # Calculate the percent difference
            difference = np.abs(real_coords - predicted_coords)
            percent_difference = (difference / real_coords) * 100

            # Extract x and y percent difference
            x_percent_diff.append(round(percent_difference[0][0], 1))
            y_percent_diff.append(round(percent_difference[0][1], 1))

            # Print the real and predicted coordinates, as well as the percent difference
            print(f'Real: {real_coords}, Predicted: {predicted_coords}, Percent Difference: {percent_difference}%')

    # Calculate statistics
    x_stats = {
        'Min': np.min(x_percent_diff),
        'Max': np.max(x_percent_diff),
        'Mean': np.mean(x_percent_diff),
        'Q1': np.percentile(x_percent_diff, 25),
        'Q3': np.percentile(x_percent_diff, 75)
    }
    y_stats = {
        'Min': np.min(y_percent_diff),
        'Max': np.max(y_percent_diff),
        'Mean': np.mean(y_percent_diff),
        'Q1': np.percentile(y_percent_diff, 25),
        'Q3': np.percentile(y_percent_diff, 75)
    }

    # Print the statistics
    print('X Percent Difference Statistics:')
    for stat, value in x_stats.items():
        print(f'{stat}: {value}')
    print('')

    print('Y Percent Difference Statistics:')
    for stat, value in y_stats.items():
        print(f'{stat}: {value}')
    print('')

    # Create a window to display the statistics
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.boxplot(x_percent_diff)
    plt.title('X Percent Difference')
    plt.subplot(1, 2, 2)
    plt.boxplot(y_percent_diff)
    plt.title('Y Percent Difference')
    plt.show()

if (__name__ == '__main__'):
    train_model()
    eval_model()