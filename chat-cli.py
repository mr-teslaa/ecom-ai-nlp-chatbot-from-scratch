# chat-cli.py
import sys

sys.path.append(".")

from app.core_ai.utils import get_response

bot_name = "Ava"
print(f"\n\n{bot_name} is ready to chat! (type 'quit' to exit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = get_response(user_input)
    print(f"{bot_name}: {response}")
