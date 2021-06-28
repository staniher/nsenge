import numpy as np
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)

# page index principale qui appelle index.html. En fait, pour utiliser le modele on fera 127.0.0.1:5000 dans le navigateur
@app.route('/')
def home():
    return render_template('index.html')
#La methode predict est appelee dans  action="{{ url_for('predict')}}" du formulaire html
@app.route('/predict',methods=['POST'])
def predict():
    #import joblib
    model=tf.keras.models.load_model('KCA_ML_Deployment_LSTM.h5') #J'utilise joblib.load pour charger notre modele sauvegarde avec joblib.dump()
    '''
    For rendering results on HTML GUI
    '''
    text = [x for x in request.form.values()]
    #final_features = [np.array(int_features)]
    # Vocabulary size
    voc_size=5000
    sent_length=100
    onehot_reprtest=[one_hot(words,voc_size)for words in text] 
    embedded_docstest=pad_sequences(onehot_reprtest,padding='pre',maxlen=sent_length)
    test_model=np.array(embedded_docstest)
    y_predtest=model.predict_classes(test_model) 
    
    #We get polarity value
    polarity=y_predtest[0]
    polarity_predict=''
    if (polarity == 2):
        polarity_predict='Positive'
    if(polarity == 0):
        polarity_predict='Negative'
    if(polarity == 1):
        polarity_predict='Neutral'
        
    #We get probability value
    import numpy
    y_predtestProba=model.predict_proba(test_model)
    y_predtestProba =numpy.max(y_predtestProba)
    y_predtestProba=y_predtestProba*100
    y_predtestProba=round(y_predtestProba,2)
    #y_predtestProba=y_predtestProba+'%'
    
        
    
    #prediction_text est appelee dans la page html pour afficher les donnees renderisees 
    #On devra mettre index.html dans le dossier template pour que le systeme le reconnaisse
    return render_template('index.html', polarity_prediction='Your taped text has {} opinion'.format(polarity_predict),probality_prediction='The probality to be {} is {} %'.format(polarity_predict,y_predtestProba))
  

if __name__ == "__main__":
    app.run(debug=True)