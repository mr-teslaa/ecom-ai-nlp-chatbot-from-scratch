# app/core_ai/utils.py
import random
import json
import torch
import numpy as np

from .nltk_utils import tokenize, bag_of_words
from .models import NeuralNet

# Load model artifacts once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("app/core_ai/intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

FILE = "app/core_ai/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Initialize and load model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def get_response(user_input):
    """
    Predicts intent and returns a response based on user input.
    Returns a random response from the matching intent or a fallback message.
    """
    # Tokenize and process input
    tokenized_input = tokenize(user_input)
    bow_vector = bag_of_words(tokenized_input, all_words)
    input_tensor = bow_vector.reshape(1, -1)
    input_tensor = torch.from_numpy(input_tensor).float().to(device)

    # Predict intent
    output = model(input_tensor)
    _, predicted_idx = torch.max(output, dim=1)
    predicted_tag = tags[predicted_idx.item()]

    probs = torch.softmax(output, dim=1)
    prob_value = probs[0][predicted_idx.item()].item()

    # Confidence threshold
    if prob_value > 0.75:
        for intent in intents["intents"]:
            if intent["tag"] == predicted_tag:
                return random.choice(intent["responses"])
    else:
        return "I do not understand..."
