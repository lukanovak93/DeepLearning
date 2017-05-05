from collections import Counter
import numpy as np

class ConversationPreprocessor:
    def __init__(self, path, batch_size, sequence_length):
        self.path = path
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def preprocess(self):
        with open(self.path, 'r') as f:
            data = f.read()
            cnt = Counter()
            for c in data:
                cnt[c] += 1
        self.sorted_chars = [x[0] for x in cnt.most_common()]
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
        self.id2char = {k:v for v,k in self.char2id.items()}
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        return np.array(list(map(self.char2id.get, sequence)))

    def decode(self, sequence):
        return np.array(list(map(self.id2char.get, sequence)))

    def create_minibatches(self):
        self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_length))
        self.current_batch = -1

        self.batches = []
        for i in range(self.num_batches):
            lx = i * self.batch_size * self.sequence_length
            batch_x = []
            batch_y = []
            for j in range(self.batch_size):
                l = lx + j * self.sequence_length
                u = lx + (j + 1) * self.sequence_length
                batch_x.append(self.x[l : u])
                batch_y.append(self.x[l+1 : u+1])
            self.batches.append((np.array(batch_x), np.array(batch_y)))

    def next_minibatch(self):
        new_epoch = False
        self.current_batch += 1
        if self.current_batch == self.num_batches:
            self.current_batch = 0
            new_epoch = True

        batch_x, batch_y = self.batches[self.current_batch]
        return new_epoch, batch_x, batch_y


# TEST_DRIVE
# proc = ConversationPreprocessor('Try.txt', 2, 5)
# proc.preprocess()
# proc.create_minibatches()
# for x, y in proc.batches:
#     for i in range(len(x)):
#         print('x: ' + str(proc.decode(x[i])))
#         print('y: ' + str(proc.decode(y[i])))
