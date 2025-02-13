from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random

# Load model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

app = Flask(__name__)

# Store chat history (Optional: can use a database)
chat_history = []

@app.route("/api/get", methods=["GET"])
def get_chat_history():
    """Fetch chat history"""
    return jsonify({"history": chat_history})

@app.route("/api/post", methods=["POST"])
def post_chat():
    """Receive user message and return chatbot response"""
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    response = get_response(user_message)
    
    # Save message & response
    chat_history.append({"user": user_message, "bot": response})

    return jsonify({"sender":"AI" ,"text":response})

def get_response(user_input):
    """Match user input with predefined intents, otherwise use DialoGPT"""
    for intent in intents["intents"]:
        if user_input.lower() in [pattern.lower() for pattern in intent["patterns"]]:
            return random.choice(intent["responses"])
    
    return get_Chat_response(user_input)  # Fallback to DialoGPT

def get_Chat_response(text):
    """Generate response using DialoGPT if no intent matches"""
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
