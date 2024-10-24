from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    return get_Chat_response(msg)

def get_Chat_response(user_input):
    # Encode user input and append EOS token
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Check if chat history exists and append the new input to it, otherwise start with new input
    global chat_history_ids
    if 'chat_history_ids' in globals():
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Generate a response while limiting the total chat history to 1000 tokens
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the output response and skip special tokens
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_response

if __name__ == '__main__':
    app.run(debug=True)
