import torch.nn as nn


class NeuralNet(nn.Module):
    """
    A basic 3-layer feed-forward neural network for intent classification.

    This architecture uses a standard linear network with ReLU activation for NLP intent matching
    by transforming tokenized input into probability distribution across defined tags.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes neural network architecture with 3 linear layers and activation function

        Args:
            input_size (int): Size of input layer (length of bag-of-words representation)
            hidden_size (int): Size of hidden layers
            output_size (int): Size of output layer (number of available intent tags)
        """
        super(NeuralNet, self).__init__()
        # Three fully connected linear layers
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

        # Applies non-linear activation between layers
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through neural network layers with activation between each stage.

        Args:
            x (torch.Tensor): Input tensor with bag-of-words representation

        Returns:
            torch.Tensor: Output tensor containing logits for each potential intent
        """
        # First layer with activation
        out = self.l1(x)
        out = self.relu(out)

        # Second layer with activation
        out = self.l2(out)
        out = self.relu(out)

        # Final layer without activation (this allows better probability interpretation in chat)
        out = self.l3(out)

        return out
