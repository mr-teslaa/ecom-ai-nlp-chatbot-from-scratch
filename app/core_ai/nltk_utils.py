import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Initialize stemmer and ensure required nltk resources are present
stemmer = PorterStemmer()
# nltk.download("punkt")  # First-time download of tokenizer resources


def tokenize(sentence):
    """
    Tokenizes a sentence into individual words, punctuation markers, and numerical entities.

    Args:
        sentence (str): Input sentence to be split into tokens

    Returns:
        List[str]: Tokenized words/punctuation marks
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Converts word into its root/stem form using Porter Stemmer algorithm.

    Args:
        word (str): Singular word in any casing or conjugation

    Returns:
        str: Normalized root form of the word (lowercased)
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    Creates a binary vector representation of which vocabulary words appear in a sentence.

    This simplified bag-of-words representation:
    - Converts words to their root form using stemming
    - Creates an array matching length of training vocabulary
    - Sets 1 for vocabulary words present in the tokenized sentence

    Args:
        tokenized_sentence (List[str]): Input sentence split into tokens
        all_words (List[str]): List of all vocabulary words from training

    Returns:
        np.ndarray: Binary array of vocabulary word presence
    """
    # Convert sentence tokens to root forms for comparison
    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    # Initialize with all zeros
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Mark vocabulary words that appear in sentence with 1.0
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0  # Word from vocabulary found in sentence

    return bag


# #  =============== Example usage 1 ===============
# print("------- Tokenization Example -------")
# a = "Hello, how are you?"
# print(tokenize(a))
# print("-------------\n\n")

# print("------- Stemming Example -------")
# words = ["organize", "organizes", "organizing", "organized"]
# print([stem(word) for word in words])
# print("-------------\n\n")

# #  =============== Example usage 2 ===============
# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print("Bag of words:", bag)
