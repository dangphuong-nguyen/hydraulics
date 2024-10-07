
from flask import Flask, render_template, request
import os, sys
from sklearn.metrics import classification_report

from src.hydraulics import Hydraulics


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    cyclenumber = int(request.form['cyclenumber']) #request.form['cyclenumber']
    print("cyclenumber:", cyclenumber)   
    
    if parameters['predict_mode'] == 'all':
        out = pred[cyclenumber+1]
    else:
        out = Hydraulics.predict(obj, cyclenumber)
    
    return f"Hello, the prediction of the valve condition for cyclenumber {cyclenumber} is {out}."



#apply test to debug the project
def test(parameters):
    obj = Hydraulics(parameters)
    
    #Train model if needed (when re_train is True or in case there is no model yet)
    x = None 
    if not os.path.exists('models/model.pkl') or parameters['re_train']:
        #Load data from files
        dfs, dps, dpf = Hydraulics.data_load(obj, nrows=parameters['n_sample'])
        #Load embeddings
        Hydraulics.embedding_load(obj, dfs, dps)
        #Feature processing
        x,y = Hydraulics.feature_processing(obj, dfs, dps, dpf)
        #Train model
        Hydraulics.train(obj,x,y)
        
    #Predict from features
    if parameters['predict_mode'] == 'all':
        if not x:
            #Load data from files
            dfs, dps, dpf = Hydraulics.data_load(obj, nrows=parameters['n_sample'])
            #Load embeddings
            Hydraulics.embedding_load(obj, dfs, dps)
            #Feature processing
            x,y = Hydraulics.feature_processing(obj, dfs, dps, dpf)
        pred = Hydraulics.predict_all(obj, x)
        print(classification_report(y, pred))
                
    print("test done!")
        

if __name__ == '__main__':
    
    #Define parameters
    parameters = {
        'predict_mode': 'all', # 'all' or 'single'
        'n_sample': 1000,
        'input_path': 'data/',
        'model': 'GradientBoosting',
        'params_model': {
            'loss': 'log_loss',
            'n_estimators': 100,
            'max_depth': 3,
            'random_state': 0,
            'learning_rate': 0.7,
            'subsample': 0.9        
        },
        're_train': False,
        're_balancing_features': True
    }
    
   
    
    #Declare Hydraulics object
    obj = Hydraulics(parameters)
    
    #Train model if needed (when re_train is True or in case there is no model yet)
    x = None 
    if not os.path.exists('models/model.pkl') or parameters['re_train']:
        #Load data from files
        dfs, dps, dpf = Hydraulics.data_load(obj)
        #Load embeddings
        Hydraulics.embedding_load(obj, dfs, dps)
        #Feature processing
        x,y = Hydraulics.feature_processing(obj, dfs, dps, dpf)
        #Train model
        Hydraulics.train(obj,x,y)
        
        
    
    
    #Predict from features
    if parameters['predict_mode'] == 'all':
        if not x:
            #Load data from files
            dfs, dps, dpf = Hydraulics.data_load(obj, nrows=parameters['n_sample'])
            #Load embeddings
            Hydraulics.embedding_load(obj, dfs, dps)
            #Feature processing
            x,y = Hydraulics.feature_processing(obj, dfs, dps, dpf)
        pred = Hydraulics.predict_all(obj, x)
        
    
            
    #Run app
    app.run(host="0.0.0.0", port=5001)
    
    