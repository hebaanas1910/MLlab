#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def mcp_neuron(inputs, weights, threshold):
    total_input = np.dot(inputs, weights)
    output = 1 if total_input >= threshold else 0
    return output


# In[3]:


def test_gate(gate_name, inputs, weights, threshold):
    print(f"\nTesting {gate_name} gate:")
    print("A B | Output")
    print("-----------")
    for input_pair in inputs:
        output = mcp_neuron(input_pair, weights, threshold)
        print(f"{input_pair[0]} {input_pair[1]} |   {output}")


# In[11]:


inputs = np.array([[0,0], [0,1], [1,0], [1,1]])

and_weights = [1, 1]
and_threshold = 2
test_gate("AND", inputs, and_weights, and_threshold)


# or_weights = [1, 1]
# or_threshold = 1
# test_gate("OR", inputs, or_weights, or_threshold)

# In[6]:


xor_weights = [1, 1]
xor_threshold = 1
test_gate("XOR (MCP - will fail)", inputs, xor_weights, xor_threshold)


# In[7]:


from sklearn.neural_network import MLPClassifier

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', max_iter=1000)
mlp.fit(X, y)

predictions = mlp.predict(X)
print("\nXOR using MLP:")
for i, pred in enumerate(predictions):
    print(f"{X[i][0]} {X[i][1]} | {pred}")


# 
