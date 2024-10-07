

import os, sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import pickle



class Hydraulics(object):

    #Initiation the class with parameters
    def __init__(self, parameters):                
        self.input_path = parameters['input_path']
        self.model = parameters['model']
        self.params_model = parameters['params_model'] 
        self.re_balancing_features = parameters['re_balancing_features']      
     
    #Load raw data from files   
    def data_load(self, skiprows=None, nrows=None):
        dfs = pd.read_csv(self.input_path + "FS1.txt", header = None, skiprows = skiprows, sep="\t", nrows=nrows)
        dps = pd.read_csv(self.input_path + "PS2.txt", header = None, skiprows = skiprows, sep="\t", nrows=nrows)
        dpf = pd.read_csv(self.input_path + "profile.txt", header = None, skiprows = skiprows, sep="\t", nrows=nrows)
        return dfs, dps, dpf
    
    #Initiate the embedding of each feature. The embedding is saved in a pickle file
    """
    On the best condition, to optimize the performance of the embedding, 
    We have to perform a deeply study on ech data field and find the best model to embed them.
    In this version of project, due to the lack of time, we have to use K-means clustering 
    to create an embedding (Note: each value of the feature is attached with a 2-D point 
    with Y-axe = 0 just to run Kmeans).
    """ 
    def embedding_init(self, dfs, dps):
        self.kmeans = dict()
        for i in dfs.dtypes.index:
            tmp = np.column_stack((np.array(dfs[i].tolist()),np.array([0]*len(dfs))))
            self.kmeans["fs_" + str(i)] = KMeans(n_clusters=10, random_state=0).fit(tmp)

        for i in dps.dtypes.index:
            tmp = np.column_stack((np.array(dps[i].tolist()),np.array([0]*len(dps))))
            self.kmeans["ps_" + str(i)] = KMeans(n_clusters=5, random_state=0).fit(tmp)
            
        with open('models/embeddings.pkl', 'wb') as file:
            pickle.dump(self.kmeans, file)
            
    #Load the embedding from a pickle file    
    def embedding_load(self, dfs, dps):
        if os.path.exists('models/embeddings.pkl'):
            with open('models/embeddings.pkl', 'rb') as file:
                self.kmeans = pickle.load(file)
        else:
            self.embedding_init(dfs, dps)
    
            
    #Feature processing with the embedding
    def feature_processing(self, dfs, dps, dpf):
        x=[]
        #Feature processing for data in FS1.txt
        for i in dfs.dtypes.index:
            tmp = np.column_stack((np.array(dfs[i].tolist()),np.array([0]*len(dfs))))
            x.append(self.kmeans["fs_" + str(i)].predict(tmp))


        #Feature processing for data in PS2.txt
        for i in dps.dtypes.index:
            tmp = np.column_stack((np.array(dps[i].tolist()),np.array([0]*len(dps))))
            x.append(self.kmeans["ps_" + str(i)].predict(tmp))
        
        x = np.column_stack(x)
        #get label from profile.txt
        y = np.array(dpf[4].tolist())
        return x,y
    
    #Train the model using chosen algorithm. The model is saved in a pickle file. 
    """
    In some condition of input data, it is necessary to balance the dataset.
    For this, we can use the imblearn library.
    """
    def train(self,x,y):
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        #Re-balancing the dataset
        if self.re_balancing_features:
            rus = RandomUnderSampler(random_state=0)
            X_train, Y_train = rus.fit_resample(X_train, Y_train)
        #Train the model
        if self.model == 'GradientBoosting':
            self.clf = GradientBoostingClassifier(**self.params_model).fit(X_train, Y_train)
        elif self.model == 'DecisionTree':
            self.clf = DecisionTreeClassifier(**self.params_model).fit(X_train, Y_train)
        #Check the performance
        y_pred = self.clf.predict(X_test)
        print(classification_report(Y_test, y_pred))
        
        #Save the model
        with open('models/model.pkl', 'wb') as file:
            pickle.dump(self.clf, file)
    
    #Predict for all raw data
    def predict_all(self, x):
        with open('models/model.pkl', 'rb') as file:
            self.clf = pickle.load(file)
            
        out = self.clf.predict(x)
        return out
    
    #Predict for one line of raw data (i.e. one specific cycle number)
    def predict(self, linenumber):
        #Load data of the specific line
        dfs = pd.read_csv(self.input_path + "FS1.txt", header = None, skiprows = linenumber-1, sep="\t", nrows=1)
        dps = pd.read_csv(self.input_path + "PS2.txt", header = None, skiprows = linenumber-1, sep="\t", nrows=1)
        
        #Load embedding
        with open('models/embeddings.pkl', 'rb') as file:
            self.kmeans = pickle.load(file)
            
        #Feature processing
        x=[]
        for i in dfs.dtypes.index:
            tmp = np.column_stack((np.array(dfs[i].tolist()),np.array([0]*len(dfs))))
            x.append(self.kmeans["fs_" + str(i)].predict(tmp))

        for i in dps.dtypes.index:
            tmp = np.column_stack((np.array(dps[i].tolist()),np.array([0]*len(dps))))
            x.append(self.kmeans["ps_" + str(i)].predict(tmp))
        
        x = np.column_stack(x)
        
        #Load model
        with open('models/model.pkl', 'rb') as file:
            self.clf = pickle.load(file)
            
        #Predict
        out = self.clf.predict(x)
        return out[0]
    
    
                