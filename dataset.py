import numpy as np
import collections

class Preprocess():

    def __init__(self, path, batch_size, sequence_length):
        self.path = path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.current_minibatch = -1


    def preprocess(self):
        f=  open(self.path, "r")
        data = f.read()
        
        # count and sort most frequent characters
        d = collections.defaultdict(int)
        for c in data:
            d[c] += 1

        
        self.sorted_chars = []
        for w in sorted(d, key=d.get, reverse=True):
            self.sorted_chars.append(w)

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars)))) 
        # reverse the mapping
        self.id2char = {k:v for v,k in self.char2id.items()}
        # convert the data to ids
        self.x = self.encode(data)


    def encode(self, sequence):
        # returns the sequence encoded as integers
        return self.oh_encode(np.array(list(map(self.char2id.get, sequence))))


    #radi ako dobije list
    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        return np.array(list(map(self.id2char.get, encoded_sequence)))

    def oh_encode(self, x):
        x_shape = list(x.shape)
        x_shape.append(71)
        X = np.zeros(x_shape)
        for i in range(71):
            X[np.arange(x_shape[0]), x] = 1
        return X


    def create_minibatches(self):
        #self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_length))
        # handling batch pointer & resetproba
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function callence_length)) # calculate the number of batches
        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?

        slice_list = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz) if len(lst[i:i+sz])==sz]
        x = np.array(slice_list(self.x, self.sequence_length))
        y = np.array(slice_list(self.x[1:], self.sequence_length))
        pairs = np.array(list(zip(x,y)))

        #da se rijesim viska
        x = np.array(list(zip(*pairs))[0])
        y = np.array(list(zip(*pairs))[1])

        # num_batches x batch_size x seq_length
        self.batches_x = np.array(slice_list(x, self.batch_size))
        self.batches_y = np.array(slice_list(y, self.batch_size))
        self.num_batches = self.batches_x.shape[0]


    def next_minibatch(self):
        self.current_minibatch +=1
        new_epoch = False
        if self.current_minibatch >= self.num_batches:
            self.current_minibatch = 0
            new_epoch = True

        batch_x = self.batches_x[self.current_minibatch]
        batch_y = self.batches_y[self.current_minibatch]

        return new_epoch, batch_x, batch_y

#pre = Preprocess('/home/katarina/Documents/faks/duboko/lab3/proba.txt', 3, 3)
#pre.preprocess()
#pre.create_minibatches()
#print(pre.next_minibatch())
#print(pre.next_minibatch())