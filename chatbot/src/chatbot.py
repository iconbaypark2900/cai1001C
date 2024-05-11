# src/chatbot.py
from chatterbot import ChatBot

def create_chatbot():
    # Create a chatbot
    chatbot = ChatBot('MyBot')
    return chatbot

def get_response(chatbot, user_input):
    # Get a response to an input statement
    response = chatbot.get_response(user_input)
    return response
