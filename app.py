from flask import Flask, request,render_template
from model import TextPreprocessor
import joblib

#Intializing flask app
app=Flask(__name__)

#Loading pretrained model pkl file
pipeline=joblib.load('spam_classification_pipeline.pkl')

@app.route('/')
def index():
    return render_template('index.html')

#Defining route for prediction
@app.route('/predict',methods=['GET','POST'])
def predict():
     if request.method == 'POST':
        #Input text
        user_msg_input = request.form['user_input'] or ''
        
        if not user_msg_input.strip():
            return render_template('index.html' ,result='not provided')

    #Predicting using pipeline
        prediction=pipeline.predict([user_msg_input])

    #Returning prediction as json
        return render_template('index.html', result= prediction[0],  msg_input = user_msg_input)
     
     else:
        return render_template('index.html')

         

#Run the app
if __name__=='__main__':
    app.run( debug=True,port=8080)
