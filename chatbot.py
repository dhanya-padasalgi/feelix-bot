#------ for the chat bot

from flask import Flask, request
import telegram
import regex as re

from telegram.ext import Updater, Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext
import transformers
from transformers import pipeline
import random
import json
#from telegram.updates import Update
global bot
global TOKEN

bot_token = "6637119559:AAEDd5pZqj2SlJrNaKvwEMQM6xaNu61xL1I"
bot_user_name = "@https://t.me/Feeli_xbot"
TOKEN = bot_token
bot = telegram.Bot(token=TOKEN)


async def start_command(update: Updater, context: CallbackContext):
    await update.message.reply_text("hello i am feelix")


def reply_message(text):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    data=(classifier(text))
    sentiment=data[0][0]['label']
    
    with open('sentiments.json', 'r') as file:
        data = json.load(file)

    # Find the intent with the specified tag
    intent = next((item for item in data["intents"] if item["tag"] == sentiment), None)
    if intent:
        responses = intent["response"]
        random_response = random.choice(responses)
        return (random_response)
    else:
        return("sorry could not process text")

async def handle_message(update: Updater, context: CallbackContext):
    text = update.message.text
    response = reply_message(text)
    await update.message.reply_text(response)
    

async def error(update: Updater, context: CallbackContext):
    print("error")


if _name=='__main_':
    app=Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler('start',start_command))

    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    app.run_polling(poll_interval=3)