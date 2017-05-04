import numpy as np
from random import shuffle

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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    score = np.dot(X[i,:], W)
    score -= np.max(score)
    p = np.exp(score) / np.sum(np.exp(score))
    loss += -1.0 * np.log(p[y[i]])
    dW += np.reshape(X[i,:].transpose(), [W.shape[0],1]) * np.reshape(p,[1, num_classes]);
    dW[:,y[i]] += -1.0 * X[i,:].transpose()
  dW /= num_train
  dW += reg * W


  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  scores -= np.reshape(np.amax(scores, axis=1),[num_train, 1])
  scores = np.exp(scores) / np.reshape(np.sum(np.exp(scores), axis = 1), [num_train,1])
  loss = -1.0 * np.log(scores[np.arange(num_train), y])
  loss = np.mean(loss)
  loss += 0.5 * reg * np.sum(W * W)
  scores[np.arange(num_train), y] -= 1.0
  dW = np.dot(X.transpose(), scores);
  dW /= num_train
  dW += reg * W

  #print dW_j.shape


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

