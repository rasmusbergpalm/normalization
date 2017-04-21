import random
import numpy as np


class ParallelReader:
    def __init__(self, source_fname, source_vocab_fname, source_delim, target_fname, target_vocab_fname, target_delim):
        self.source = Reader(source_fname, source_vocab_fname, source_delim)
        self.target = Reader(target_fname, target_vocab_fname, target_delim)
        self.samples = list(zip(self.source.padded, self.target.padded))

    def source_vocab_size(self):
        return len(self.source.vocab)

    def target_vocab_size(self):
        return len(self.target.vocab)

    def all(self):
        return map(np.array, zip(*self.samples))

    def random_batch(self, batch_size):
        sample = random.sample(self.samples, batch_size)
        return map(np.array, zip(*sample))

    def batched_iterator(self, batch_size):
        l = len(self.samples)
        for ndx in range(0, l, batch_size):
            yield self.samples[ndx:min(ndx + batch_size, l)]


class Reader:
    pad_char = "<PAD>"
    go_char = "<GO>"
    eos_char = "<EOS>"

    def __init__(self, fname, vocab_fname, delim):
        self.delim = delim
        self.vocab = self._read_vocab(vocab_fname)
        self.rvocab = {v: k for k, v in self.vocab.items()}
        self.padded = self._pad(self._read(fname))

    def idx_to_str(self, idxs: np.ndarray):
        idxs = idxs.tolist()
        outputs = []
        for row in idxs:
            outputs.append(self.delim.join([self.rvocab[idx] for idx in row if idx > 2]))
        return outputs

    def str_to_idx(self, str: str):
        def _split(string, delim):
            return list(string) if delim == "" else string.split(delim)

        tokens = [self.go_char] + _split(str.strip(), self.delim) + [self.eos_char]
        return [self.vocab[tok] for tok in tokens]

    def _pad(self, seq: list):
        seqmax = len(max(seq, key=len))
        seq = [x + [0] * (seqmax - len(x)) for x in seq]
        return np.array(seq, dtype=np.int32)

    def _read(self, fname):
        with open(fname) as f:
            lines = []
            for line in f.readlines():
                lines.append(self.str_to_idx(line))

        return lines

    def _read_vocab(self, vocab_fname):
        vocab = dict()
        for c in [self.pad_char, self.go_char, self.eos_char]:
            vocab[c] = len(vocab)

        with open(vocab_fname) as f:
            for line in f:
                vocab[line.strip('\n')] = len(vocab)

        return vocab
