from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        sample = X[i]
        scores = sample.dot(W)
        exp_scores = np.exp(scores)
        exp_sum = np.sum(exp_scores)
        loss -= np.log(exp_scores[y[i]]/exp_sum)
        
        
        for j in range(num_classes):
            if(j==y[i]):
                dW[:,y[i]] += (-1)*sample*((exp_sum - exp_scores[y[i]])/(exp_sum))
            else:
                dW[:,j] += sample*(exp_scores[j]/(exp_sum))

    
    loss = loss/num_train
    loss = loss + reg*np.sum(W*W)
    
    dW = dW/num_train
    dW = dW + 2*reg*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    scores = X.dot(W)
    exp_scores = np.exp(scores)
    
    exp_sum = np.sum(exp_scores, axis = 1)
    
    correct_scores = exp_scores[np.arange(num_train),y]
    loss_vector = -np.log(correct_scores/exp_sum)
    loss = np.sum(loss_vector)
    loss = loss/num_train
    loss += reg*np.sum(W*W)
    
    #gradients for all except the correct class
    grads = (exp_scores.T/exp_sum).T
    #gradients for the correct class
    grads[np.arange(num_train),y] = -(1 - (exp_scores[np.arange(num_train),y]/exp_sum))
    dW = X.T.dot(grads)
    
    dW /= num_train
    dW += (2*reg*W)    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
