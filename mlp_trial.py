# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

class MLP:
    def __init__(self, inodes, hnodes, onodes, learningrate=0.1):
        # this contains the architecture
        
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        
        self.inputs = np.zeros((inodes,1))
        self.hidden = np.zeros((hnodes,1))
        self.outputs = np.zeros((onodes,1))
        
        self.Bih = np.random.normal(0.0,1,(hnodes,1))
        self.Bho = np.random.normal(0.0,1,(onodes,1))
        
        self.Wih = np.random.normal(0.0,1,(hnodes,inodes))
        self.Who = np.random.normal(0.0,1,(onodes,hnodes))
        
        self.activation = lambda x : 1/(1+np.exp(-x))
        
        self.lr = learningrate

    def feedforward(self,inp):
        self.inputs = np.array(inp,ndmin=2).T
        self.hidden = self.activation(np.dot(self.Wih,self.inputs) + self.Bih)
        self.outputs = self.activation(np.dot(self.Who,self.hidden) + self.Bho)
        return self.outputs
    
    def train(self,question,answer):
        inp = np.array(question,ndmin=2).T
        hid = self.activation(np.dot(self.Wih,inp) + self.Bih)
        out = self.activation(np.dot(self.Who,hid) + self.Bho)
        
        out_error = answer-out
        hid_error = np.dot(self.Who.T, out_error)
    
        self.Who += self.lr * np.dot((out_error * out * (1.0 - out)),hid.T)
        self.Wih += self.lr * np.dot((hid_error * hid * (1.0 - hid)),inp.T)
        
       
        