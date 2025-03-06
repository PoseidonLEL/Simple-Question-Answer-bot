# Simple-Question-Answer-bot
A simple Question-Answering Bot class that uses a predetermined Neural Network structure to train itself on a dataset of Questions and their Corresponding Answers provided by the user.


## Main Methods:
    
- train_new_bot(exmaple_questions, example_answers): Initialize and train a new QA_Bot with example questions and answers.

- query(question): When a trained model is loaded in the class, use this function to ask the bot a question.

- export_model(): Export the currently loaded model and vectorizer as files to the current directory.

- import_model(): Import a trained model and vectorizer from files in the current directory.
