# src/train_chatbot.py
from chatterbot.trainers import ChatterBotCorpusTrainer

def train_chatbot(chatbot):
    # Create a new trainer for the chatbot
    trainer = ChatterBotCorpusTrainer(chatbot)

    # Train the chatbot based on the English language corpus
    trainer.train('chatterbot.corpus.english')
