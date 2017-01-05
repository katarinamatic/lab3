import numpy as np
from dataset import Preprocess


class RNN():

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate=1e-1):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        mu=0
        sigma = 1e-2
        self.U = np.random.normal(mu, sigma, (self.hidden_size, self.vocab_size))
        self.W = np.random.normal(mu, sigma, (self.hidden_size, self.hidden_size))
        self.b = np.random.normal(mu, sigma, (self.hidden_size, 1))

        self.V = np.random.normal(mu, sigma, (self.vocab_size, self.hidden_size))
        self.c = np.random.normal(mu, sigma, (self.vocab_size, 1))

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)


    @staticmethod
    def softmax(s):
        # batch x sequence x out
        exps = np.exp(s)
        return exps / np.sum(exps, axis=2, keepdims=True)


    @staticmethod
    def loss(y, o):
        return np.log(np.sum(np.exp(o), axis=1)) - np.sum(y * o, axis=1)


    @staticmethod
    def rnn_step_forward(x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        lin = (np.dot(h_prev, W) + np.dot(x, U.T)) + b.T
        h_current = np.tanh(lin)
        cache = (h_prev, h_current, x)

        # return the new hidden state and a tuple of values needed for the backward step

        return h_current, cache

    @staticmethod
    def rnn_forward( x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        seq_len = x.shape[1]

        shape_of_h = list(h0.shape)
        #za svaki u seq(za svaki timestamp)
        shape_of_h.insert(1, seq_len + 1)
        h = np.zeros(shape_of_h)
        #na prvo mjesto stavi h0
        h[:,0,:] = h0
        cache = []

        
        for t in range(seq_len):
            h_t, cache_t = RNN.rnn_step_forward(x[:,t,:], h[:,t,:], U, W, b)
            h[:, t+1,:] = h_t
            cache.append(cache_t)

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
        return h, cache


    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass
        h_prev = cache[0]
        h = cache[1]
        x = cache[2]
        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative oh.shapef the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters
        dtan_h = 1 - h**2
        dLdh_prev = np.dot(grad_next, np.dot(dtan_h, self.W))
        dLdU = np.dot(grad_next, np.dot(dtan_h, x))
        dLdW = np.dot(grad_next, np.dot(dtan_h, h_prev))
        dLdb = np.sum(np.dot(grad_next, dtan_h), axis=0, keepdims=True)


        return dLdh_prev, dLdU, dLdW, dLdb


    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        dLdU = np.zeros((self.hidden_size, self.vocab_size))
        dLdW = np.zeros((self.hidden_size, self.hidden_size))
        dLdb = np.zeros((self.hidden_size, 1))
        dLdh_prev = 0
        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?
        for t in reversed(range(self.sequence_length-1)):
            dLdh_t = dh[:,t,:] + dLdh_prev
            dLdh_prev, dLdu, dLdw, dLdb = self.rnn_step_backward(dLdh_t, cache[t])

            dLdU += np.clip(dLdu, -5, 5)
            dLdW += np.clip(dLdw, -5, 5)
            dLdb += np.clip(dLdb, -5, 5)

        return dLdU, dLdW, dLdb

    @staticmethod
    def output(h, V, c):
        # Calculate the output probabilities of the network
        batch_size = h.shape[0]
        time_size = h.shape[1]
        vocab_size = V.shape[0]
        output = np.zeros((batch_size, time_size - 1, vocab_size))
        for i in range(1, time_size):
            o_i = np.dot(h[:, i, :], V.T) + c.T
            output[:, i - 1, :] = o_i
        return output 

    def output_loss_and_grads(self, h, V, c, y):
        # Calculate the loss of the network for each of the outputs
        
        # h - hidden states of the network for each timestep. 
        #     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a tensor of dimension 
        #     batch_size x sequence_length x vocabulary size -

        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h
        output = RNN.output(h, V, c)
        soft_output = RNN.softmax(output)
        loss = RNN.loss(y, output)
        print("loss: ", np.mean(loss))

        dLdh = np.dot((soft_output-output), V)
        dLdV = 0
        dLdc = 0
        dLdoutput = soft_output - y

        for t in range(self.sequence_length):
            dLdV += np.clip(np.dot(dLdoutput[:,t,:].T, h[:,t+1,:]), -5, 5)
            dLdc += np.clip(np.sum(dLdoutput[:,t,:], axis=0, keepdims=True), -5, 5)

        return loss, dLdh, dLdV, dLdc


    def update(self, dU, dW, db, dV, dc, theta = 1e-8):
        for w, grad, m in zip(( self.U, self.W, self.b, self.V, self.c),(dU, dW, db.T, dV, dc.T),(self.memory_U, self.memory_W, self.memory_b, self.memory_V, self.memory_c)):
            m += grad * grad
            w += -(self.learning_rate / (theta + np.sqrt(m)) * grad)

    
    def step(self, h0, x, y):
        h, cache = RNN.rnn_forward(x, h0, self.U, self.W, self.b)
        loss, dLdh, dLdV, dLdc = self.output_loss_and_grads(h, self.V, self.c, y)
        dLdU, dLdW, dLdb = self.rnn_backward( dLdh, cache)
        self.update(dLdU, dLdW, dLdb, dLdV, dLdc)

        return loss, h[:,-1,:]



def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100):
    
    vocab_size = len(dataset.sorted_chars)
    net = RNN(hidden_size, sequence_length, 71, learning_rate=1e-3)

    current_epoch = 0 
    batch = 0

    h0 = np.zeros((dataset.batch_size, hidden_size))

    average_loss = 0

    while current_epoch < max_epochs: 
        e, x_oh, y_oh = dataset.next_minibatch()
        
        if e: 
            current_epoch += 1
            h0 = np.zeros((dataset.batch_size, hidden_size))
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = net.step(h0, x_oh, y_oh)

        if batch % sample_every == 0: 
            # run sampling (2.2)
            pass
        batch += 1

    

dataset = Preprocess('/home/katarina/Documents/faks/duboko/lab3/selected_conversations.txt', 100, 30)
dataset.preprocess()
dataset.create_minibatches()
"""
new_epoch, batch_x, batch_y = dataset.next_minibatch()
rnn = RNN(100, 30, 71, 1e-3)
h0 = np.zeros((100, rnn.hidden_size))
h, cache = RNN.rnn_forward(batch_x, h0, rnn.U, rnn.W, rnn.b)
o = RNN.output(h, rnn.V, rnn.c)
rnn.rnn_backward(h, cache)
rnn.output_loss_and_grads(h, rnn.V, rnn.c, batch_y)self,
rnn.update(rnn.U, rnn.W, rnn.b, rnn.V, rnn.c)"""

run_language_model(dataset, 10)