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

dataset_root = 'datasets/score_tracker'
model_root = 'models/'
model_name= 'score_tracker.pth'

num_epochs = 400

class ScoreTracker:
    def __init__(self, model_name=model_name):
        self.model = ScoreTrackerModel()
        self.model.load_state_dict(torch.load(model_root + model_name))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def get_score(self, image):
        image_tensor = self.transform(image.convert('L')).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.max(output, 1)[1][0].item()

        if output == 10:
            return 0
        else:
            return output

class ScoreTrackerModel(nn.Module):
    def __init__(self):
        super(ScoreTrackerModel, self).__init__()

        # Define the CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_input_size = 64 * 28 * 28

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 11)  # Output size 11 for the digit confidences
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ScoreTrackerDataset(Dataset):
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
                additional_values.append(int(value))
        return additional_values

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # Load the image
        image = Image.open(image_path).convert('L')

        # Convert the image to a Tensor
        image = torchvision.transforms.ToTensor()(image)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Get the additional value associated with the image
        value = self.additional_values[index]
        value = torch.tensor(value, dtype=torch.long)

        return image, value
    
def train_model(model_name=model_name, load_model=False):
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ScoreTrackerDataset(dataset_root + '/train', transform, dataset_size=72)    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Define model
    model = ScoreTrackerModel()
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

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

        if epoch % 25 == 0:
            torch.save(model.state_dict(), model_root + f'score_tracker_{epoch}.pth')

        if (average_loss < lowest_average_loss):
            lowest_average_loss = average_loss
            torch.save(model.state_dict(), model_root + 'score_tracker_lowest_loss.pth')

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
    dataset = ScoreTrackerDataset(dataset_root + "/test", transform, dataset_size=30)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the trained model
    model = ScoreTrackerModel()
    model.load_state_dict(torch.load(model_root + model_name))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Get predicted classes
        _, predicted_classes = torch.max(outputs, 1)
        
        # Count correct predictions
        correct_predictions += (predicted_classes == labels).sum().item()
        total_predictions += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions * 100.0

    print(f'Average Loss: {average_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')

if (__name__ == '__main__'):
    train_model()
    eval_model()
    eval_model('score_tracker_lowest_loss.pth')

    for i in range(num_epochs):
        if i%25 == 0:
            eval_model(f'score_tracker_{i}.pth')