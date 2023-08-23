 #------ for the chat bot

import re
from flask import Flask, request
import telegram
from telebot.credentials import bot_token, bot_user_name,URL
from telebot.bot_engine import generate_reply
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

global bot
global TOKEN
TOKEN = bot_token
bot = telegram.Bot(token=TOKEN)

# start the flask app
app = Flask(__name__)

@app.route('/{}'.format(TOKEN), methods=['POST'])
def respond():
    # retrieve the message in JSON and then transform it to Telegram object
    update = telegram.Update.de_json(request.get_json(force=True), bot)

    chat_id = update.message.chat.id
    msg_id = update.message.message_id

    # Telegram understands UTF-8, so encode text for unicode compatibility
    text = update.message.text.encode('utf-8').decode()
    # for debugging purposes only
    print("got text message :", text)

    # the first time you chat with the bot AKA the welcoming message
    if text == "/start":
        # print the welcoming message
        bot_welcome = """
        Welcome to Feelix, your virtual companion.
        """
        # send the welcoming message
        bot.sendMessage(chat_id=chat_id, text=bot_welcome, reply_to_message_id=msg_id)

    elif text == "/help":
        bot_welcome = """
        I am a Feelix! Please type something so I can respond!
        """
        bot.sendMessage(chat_id=chat_id, text=bot_welcome, reply_to_message_id=msg_id)

    elif text == "/custom":
        bot_welcome = """
       'This is a custom command!'
       """
        bot.sendMessage(chat_id=chat_id, text=bot_welcome, reply_to_message_id=msg_id)


    else:
        try:
            # clear the message we got from any non alphabets
            text = re.sub(r"\W", "_", text)
            # integrate model to chat bot 
            #reply comes as "reply_message"
            reply_message=generate_reply(text)
            bot.sendMessage(chat_id=chat_id, text=reply_message , reply_to_message_id=msg_id)
            
        except Exception:
            # if things went wrong
            bot.sendMessage(chat_id=chat_id, text="There was a problem in the name you used, please enter different name", reply_to_message_id=msg_id)

    return 'ok'


    @app.route('/set_webhook', methods=['GET', 'POST'])
    def set_webhook():
        s = bot.setWebhook('{URL}{HOOK}'.format(URL=URL, HOOK=TOKEN))
    if s:
        return "webhook setup ok"
    else:
        return "webhook setup failed"

@app.route('/')
def index():
    return '.'

#To check if the msg is sent in grp chat or pvt chat
async def handle_message (update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
         if bot_user_name in text:
             new_text: str = text.replace(bot_user_name, '').strip()
             response: str = respond (new_text)
         else:
             return 
    else:
         response: str = respond(text)
    print('Bot:', response)
    await update.message.reply_text(response)

#Handle errors
async def error (update: Update, context: ContextTypes.DEFAULT_TYPE):
     print(f'Update {update} caused error {context.error}')

bot.add_handler(MessageHandler(filters.text, handle_message))
bot.add_error_handler(error)


if __name__ == '__main__':
    # note the threaded arg which allow
    # your app to have more than one thread
    app.run(threaded=True)