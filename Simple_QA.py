from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
import joblib

#Small warning class for custom warnings under certain conditions
class QA_Bot_Warning(UserWarning):
    pass


#Very simple Question -> Answer bot that uses an MLP Neural Network to predict the answer to a given question.
class QA_Bot:

    # Declare initial class data
    def __init__(self): 

        self.questions = None
        self.answers = None
        self.__class_vectorizer = None
        self.__encoded_questions = None
        self.__fitted_Network = None
        self.__status = 0



    # Initialize new QA_Bot with example questions and answers.

      # Parameters:
        # exmaple_questions (list): A list of questions used for training the model.
        # example_answers (list): A list of corresponding answers to the questions.
        # use_neural_net (boolean): Boolean value on whether to use Neural Net or Decision Tree (defaults to True).
    def train_new_bot(self, exmaple_questions, example_answers, use_neural_net=True):

        self.questions = exmaple_questions
        self.answers = example_answers
        self.__class_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.__encoded_questions = self.__class_vectorizer.fit_transform(self.questions)
        self.__fitted_Network = MLPClassifier(max_iter=100000, 
                                              activation='relu', 
                                              hidden_layer_sizes=(40,40,40)).fit(self.__encoded_questions,self.answers) if use_neural_net else DecisionTreeClassifier(max_depth=100).fit(self.__encoded_questions,self.answers)
        self.__status = 1




    # Predict the answer to a given question.

      # Parameters:
        # question (str): A question to predict the answer for.

      # Returns:
        # string: The predicted answer.
    def query(self, question):

        if(self.__status==1):
            return self.__fitted_Network.predict( self.__class_vectorizer.transform([question]))[0]
        else: 
            return warnings.warn("No model has been trained yet. Import a model or use train_new_bot() to make one.", QA_Bot_Warning)




    # Export the trained model and vectorizer to files.
    # The trained model and vectorizer are saved to files named "model_classifier.mlmodel" and "model_encoder.pkl" respectively.
    def export_model(self):

        if(self.__status==1):
            joblib.dump(self.__fitted_Network, "model_classifier.mlmodel")
            joblib.dump(self.__class_vectorizer, "model_encoder.pkl")
        else:
            return warnings.warn("No model has been trained yet. Import a model or use train_new_bot() to make one.", QA_Bot_Warning)




    # Import the trained model and vectorizer from files in the current directory

      # Parameters:
        # model_file_name (str): Name of the model file.
        # encoder_file_name (str): Name of the encoder file.
    def import_model(self, model_file_name, encoder_file_name):

        self.__fitted_Network = joblib.load(model_file_name)
        self.__class_vectorizer = joblib.load(encoder_file_name)
        self.__status = 1

