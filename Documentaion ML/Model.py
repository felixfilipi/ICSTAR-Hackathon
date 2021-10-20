import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox

class Logistic_Regression() :
    
    def __init__(self) : 
        self.learning_rate = 0.01 
        self.epoch = 2000 
       
    def update_weights(self) :           
        A = 1/(1+np.exp(-(self.X.dot(self.W) + self.b)))
               
        tmp = (A-self.Y.T)        
        tmp = np.reshape(tmp,self.m)        
        dW = np.dot(self.X.T,tmp)/self.m          
        db = np.sum(tmp)/self.m 
            
        self.W = self.W - self.learning_rate * dW   
        self.b = self.b - self.learning_rate * db
          
        return self
       
    def fit(self, X, Y) :        
            
        self.m, self.n = X.shape   
              
        self.W = np.zeros(self.n)        
        self.b = 0        
        self.X = X        
        self.Y = Y
          
        for i in range(self.epoch) :       
            self.update_weights() 
        return self  
    
    def predict(self, X) :    
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
        Y = np.where( Z > 0.5, 1, 0 )        
        return Y
    
ds = pd.read_excel("Dataset.xlsx")
ds = ds[['Rating','Total Reviews','Distance','Prediction']]
ds = ds.dropna()

feature = ds.iloc[:,:-1]
target = ds.iloc[:,-1:]

target = np.array(target)
target = target.reshape(-1)

model = Logistic_Regression()
model.fit(feature, target)

def close_window():
    filename = a1.get()
    test = pd.read_excel(filename)
    test = test[['Rating','Total Reviews','Distance']]
    test = test.dropna()
    result=model.predict(test)
    with open("Output.txt", "w") as text_file:
        text_file.write(str(result))
    window.destroy()

window = tk.Tk()
window.title("Prediction")
window.geometry('200x200')

#define the label
a = tk.Label(window, text="File Name = ")
a.grid(row = 0,column = 0)
a1 = tk.Entry(window)
a1.grid(row = 0, column = 1)

button = tk.Button(window,text="Enter",bg="#000",fg="#fff", command = close_window)
button.grid(row=2, column = 1)

window.mainloop()

