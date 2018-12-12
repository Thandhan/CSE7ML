import numpy as np 
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)      # Features ( Hrs Slept, Hrs Studied) 
y = np.array(([92], [86], [89]), dtype=float)       # Labels(Marks obtained) 
X = X/np.amax(X,axis=0) # Normalize 
y = y/100 
def sigmoid(x):    
    return 1/(1 + np.exp(-x)) 
def sig_grad(x):    
    return x * (1 - x) 
epoch=1000 
lr=0.2 
input_neurons =2 
hidden_neurons =3 
output_neurons =1 # Weight and bias - Random initialization 
wh=np.random.uniform(size=(input_neurons,hidden_neurons)) 
bh=np.random.uniform(size=(1,hidden_neurons)) 
wout=np.random.uniform(size=(hidden_neurons,output_neurons)) 
bout=np.random.uniform(size=(1,output_neurons)) 
for i in range(epoch):    #Forward Propogation    
    hidip=np.dot(X,wh) + bh     
    hidact =sigmoid(hidip)     
    outip=np.dot(hidact,wout) + bout    
    output = sigmoid(outip)    #Backpropagation    
    Errout = y-output     
    outgrad = sig_grad(output)    
    d_output = Errout* outgrad     # Error at Hidden later    
    Errhid = d_output.dot(wout.T)     
    hidgrad = sig_grad(hidact)     
    d_hidden = Errhid * hidgrad    #Update weight    
    wout += hidact.T.dot(d_output) *lr         
    wh += X.T.dot(d_hidden) *lr 
print(wh) 
print(wout)
print("Normalized Input: \n" + str(X)) 
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)