from flask import Flask
from SemanticSearchBot import SemanticSearchBot

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!!!"

if __name__=='__main__':
    print("Welcome to Sematic Search. You can get the relevant answers to your queries");
    print("===================================================");
    searchBot = SemanticSearchBot()
    searchBot.initialize_chatbot()
    print("===================================================");
    print("Thank you for choosing Sematic Search chatbot");
