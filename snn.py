import numpy as np
X=np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1],
            [1,0,1],
            [1,1,1]])
Y=np.array([[0],
            [1],
            [1],
            [0],
            [1],
            [0]])
np.random.seed(100)
w1=2*np.random.random((3,5))-1
w2=2*np.random.random((5,7))-1
w3=2*np.random.random((7,5))-1
w4=2*np.random.random((5,1))-1

def sigmoid(X):
    return 1/(1+np.exp(-X))
def derivative(X):
    return X*(1-X)
def mse(Y,ycap):
    return ((Y-ycap)**2).mean()

w=[np.array(w1),np.array(w2),np.array(w3),np.array(w4)]
def train(X,Y,w,iter,conv=0.00000001):
    W1=w[0]
    W2=w[1]
    W3=w[2]
    W4=w[3]
    perr=0
    j=0
    for i in range(iter):
        l1=sigmoid (X.dot(W1))
        l2=sigmoid (l1.dot(W2))
        l3=sigmoid (l2.dot(W3))
        l4=sigmoid (l3.dot(W4))
        cerr=mse(Y,l4)
        diff=abs(perr-cerr)
        if diff<=conv:
            print("Training completed after ", i+1,"iterations.")
            j=1
            break
        if i%250==0:
            print("Current Error at ",i+1 ,"iteration :",cerr)
        e4=Y-l4
        delta4=e4*derivative(l4)
        e3=delta4.dot(W4.T)
        delta3=e3*derivative(l3)
        e2=delta3.dot(W3.T)
        delta2=e2*derivative(l2)
        e1=delta2.dot(W2.T)
        delta1=e1*derivative(l1)
        W1+=X.T.dot(delta1)
        W2+=l1.T.dot(delta2)
        W3+=l2.T.dot(delta3)
        W4+=l3.T.dot(delta4)
        perr=cerr
    if j==0:
            print("Training not completed !!!")
    return[W1,W2,W3,W4]
