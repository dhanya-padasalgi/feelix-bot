 #------ for the chat bot

from flask import Flask, request
import telegram
import regex as re

from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
global bot
global TOKEN

bot_token = "6637119559:AAEDd5pZqj2SlJrNaKvwEMQM6xaNu61xL1I"
bot_user_name = "@Feeli_xbot"
TOKEN = bot_token
bot = telegram.Bot(token=TOKEN)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPES):
    text = update.message.text
    response = reply_message(text)
    await update.message.reply_text(response)
    

async def error(update: Update, context: ContextTypes.DEFAULT_TYPES):
    print("error")


if __name__=='__main__':
    app=Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler('start',start_command))

    app.add_handler(MessageHandler(filters.TEXT, handel_message))

    app.run_polling(poll_interval=3)
