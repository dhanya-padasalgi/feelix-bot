# # start the flask app
# app = Flask(__name__)

# @app.route('/{}'.format(TOKEN), methods=['POST'])
# def respond():
#    # retrieve the message in JSON and then transform it to Telegram object
#    update = telegram.Update.de_json(request.get_json(force=True), bot)

#    chat_id = update.message.chat.id
#    msg_id = update.message.message_id

#    # Telegram understands UTF-8, so encode text for unicode compatibility
#    text = update.message.text.encode('utf-8').decode()  #input
#    # for debugging purposes only
#    print("got text message :", text)
#    # the first time you chat with the bot AKA the welcoming message
#    if text == "/start":
#        # print the welcoming message
#        bot_welcome = """
#        Welcome to Feelix, your virtual companion.
#        """
#        # send the welcoming message
#        bot.sendMessage(chat_id=chat_id, text=bot_welcome, reply_to_message_id=msg_id)


#    else:
#        try:
#            # clear the message we got from any non alphabets
#            text = re.sub(r"\W", "_", text) #input
#            # integrate model to chat bot 
#            #reply comes as "reply_message"
#            reply_message=reply_message(text)
#            bot.sendMessage(chat_id=chat_id, text=reply_message , reply_to_message_id=msg_id)
           
#        except Exception:
#            # if things went wrong
#            bot.sendMessage(chat_id=chat_id, text="There was a problem in the name you used, please enter different name", reply_to_message_id=msg_id)

#    return 'ok'


# from transformers import pipeline

# import json
# import random

# def reply_message(text):
#     classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
#     data=(classifier(text))
#     sentiment=data[0][0]['label']
    
#     with open('sentiments.json', 'r') as file:
#         data = json.load(file)

#     # Find the intent with the specified tag
#     intent = next((item for item in data["intents"] if item["tag"] == sentiment), None)
#     if intent:
#         responses = intent["response"]
#         random_response = random.choice(responses)
#         return (random_response)
#     else:
#         return("sorry could not process text")


# @app.route('/set_webhook', methods=['GET', 'POST'])
# def set_webhook():
#    s = bot.setWebhook('{URL}{HOOK}'.format(URL=URL, HOOK=TOKEN))
#    if s:
#        return "webhook setup ok"
#    else:
#        return "webhook setup failed"

# @app.route('/')
# def index():
#     return '.'
# if __name__ == '__main__':
#     # note the threaded arg which allow
#     # your app to have more than one thread
#     app.run(threaded=True)