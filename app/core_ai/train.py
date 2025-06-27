import json  # For reading intent configuration
import torch  # PyTorch framework for neural network implementation
import numpy as np  # For numeric manipulation
import torch.nn as nn  # Neural network modules
from torch.utils.data import Dataset, DataLoader  # Data handling utilities
from nltk_utils import tokenize, stem, bag_of_words  # NLP processing functions
from models import NeuralNet  # Intent classifier neural network architecture

# Load all training intents from configuration file
with open("./intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)


# Initialize storage containers
all_words = []  # Unique stemmed words from patterns
tags = []  # Unique intent tags
xy = []  # Pattern-tag pairs for training

# Process each intent pattern for training
for intent in intents["intents"]:
    # Accumulate intent tags
    tag = intent["tag"]
    if tag not in tags:
        tags.append(tag)

    for pattern in intent["patterns"]:
        # Tokenize and stem each pattern
        word_tokens = tokenize(pattern)
        stemmed_words = [stem(word) for word in word_tokens]
        all_words.extend(stemmed_words)  # Collect stemmed words for vocabulary
        xy.append((stemmed_words, tag))  # Store processed pattern with tag

# Filter out irrelevant characters for model efficiency
ignore_words = ["?", "!", ".", ",", "'s", "'m", "'re", "'ll", "'ve", "'d", "'t"]

# Create final vocabulary list with unique, stemmed words
all_words = sorted(set([w for w in all_words if w not in ignore_words]))

# Final tag list in lex order
tags = sorted(set(tags))

# Create training arrays
X_train = []  # Features
Y_train = []  # Labels

# Convert each pattern to numerical BOW representation
for pattern_sentence, tag in xy:
    # Generate bag-of-words vector for pattern
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)  # Add to feature matrix

    # Convert string tag to numerical index
    label = tags.index(tag)
    Y_train.append(label)  # Add to label array

# Convert training data to numpy arrays for PyTorch compatibility
X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


# Define neural network hyperparameters
batch_size = 8  # Size of training batches
hidden_size = 8  # Size of hidden layers
output_size = len(tags)  # Number of output classes
input_size = len(X_train[0])  # Size of input based on vocabulary
learning_rate = 0.001  # Learning rate for optimizer
num_epochs = 1000  # Total training iterations

# Create dataset instance for PyTorch
dataset = ChatDataset()

# Windows compatibility: must have num_workers=0
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

# Determine and report available training device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize network with appropriate dimensions
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Set loss function and optimizer
# CrossEntropyLoss handles raw logits and label indices
criterion = nn.CrossEntropyLoss()
# Adam optimizer with learning rate parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Begin training process
for epoch in range(num_epochs):
    for words, labels in train_loader:
        # Move data to available processing hardware
        words = words.to(device)
        labels = labels.to(device)

        # Provide samples to model and calculate error
        outputs = model(words)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss

        # Reset gradients for backpropagation
        optimizer.zero_grad()
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    # Print progress at regular interval
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Report final metrics
print(f"Final loss: {loss.item():.4f}")


# Save model assets for inference
data = {
    "model_state": model.state_dict(),  # Network weights
    "input_size": input_size,  # Vocabulary size
    "hidden_size": hidden_size,  # Internal learning capacity
    "output_size": output_size,  # Number of classes/intents
    "all_words": all_words,  # Processed vocabulary list
    "tags": tags,  # Intent tag references
}
FILE = "data.pth"
torch.save(data, FILE)
print(f"Training complete. Model saved to {FILE}")
