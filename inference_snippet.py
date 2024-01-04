from functools import partial
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Alphabet helpers
alphabet = "NACGT"
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}
dna2int.update({"pad": 0})
int2dna.update({0: "<pad>"})

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

# some config
LSTM_HIDDEN = 128
LSTM_LAYER = 1
batch_size = 32
learning_rate = 0.001
epoch_num = 300


class MyDataset(torch.utils.data.Dataset):
    """Dataset class for dataloaders"""

    def __init__(self, lists, labels) -> None:
        self.lists = lists
        self.labels = labels

    def __getitem__(self, index):
        return torch.tensor(self.lists[index]), self.labels[index]

    def __len__(self):
        return len(self.lists)


class PadSequence:
    """PadSequence Class for padding in batches"""

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=dna2int["pad"]
        )
        sequences_padded = torch.tensor(sequences_padded)
        lengths = torch.tensor([len(x) for x in sequences])
        labels = torch.tensor([x[1] for x in sorted_batch])

        return sequences_padded, lengths, labels


class CpGPredictor(torch.nn.Module):
    """Simple model that uses a LSTM to count the number of CpGs in a sequence"""

    def __init__(self):
        """Initialization"""
        super(CpGPredictor, self).__init__()
        self.lstm = torch.nn.LSTM(6, LSTM_HIDDEN, LSTM_LAYER, batch_first=True)
        self.classifier = torch.nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x, x_length):
        """ChatGPT and google helped a lot in this one phew"""
        x_pack = pack_padded_sequence(
            x, x_length, batch_first=True, enforce_sorted=True
        )
        x_padded, _ = pad_packed_sequence(x_pack, batch_first=True)
        output, _ = self.lstm(x_padded)
        output = output[:, -1, :]
        logits = self.classifier(output)
        return logits


class Inference:
    """Inference pipeline for predicting the CG count for variable length sequence strings"""

    def __init__(self) -> None:
        """Initialize"""
        self.model = torch.load("trained_model/model_variable_len_sequence.pt")

    def count_cpgs(self, seq: str) -> int:
        """Dynamically counts the number of CGs in the DNA sequence
        Args:
            seq (str): dna sequence string
        Returns:
            int: number of cg counts
        """
        cgs = 0
        for i in range(0, len(seq) - 1):
            dimer = seq[i : i + 2]
            if dimer == "CG":
                cgs += 1
        return cgs

    def to_categorical(self, y, num_classes):
        """1-hot encodes a tensor"""
        return np.eye(num_classes, dtype="uint8")[y]

    def convert_seq_to_desired_padding(self, sequence, desired_length):
        """Padding till desired length for sequences"""
        pad_right = desired_length - len(sequence)
        sequence.extend([dna2int["pad"]] * pad_right)
        return sequence

    def inference_for_single_sequence(self, sequence, maxlen=128):
        """Returns the onehot encoded sequence with padding and CG counts

        Args:
            sequence (list): DNA sequence string list
            maxlen (int): maximum length of sequence for padding

        Returns:
            X_dna_seqs_onehot_padding: onehot encoded tensor
            y_dna_seqs: CG count tensor
        """
        X_dna_seqs = sequence
        temp = ["".join(intseq_to_dnaseq(x)) for x in X_dna_seqs]
        y_dna_seqs = [self.count_cpgs(x) for x in temp]
        X_dna_seqs_padding = [
            self.convert_seq_to_desired_padding(x, desired_length=maxlen)
            for x in X_dna_seqs
        ]
        X_dna_seqs_onehot_padding = [
            self.to_categorical(x, num_classes=6) for x in X_dna_seqs_padding
        ]
        X_dna_seqs_onehot_padding = torch.tensor(
            X_dna_seqs_onehot_padding, dtype=torch.float
        )
        y_dna_seqs = torch.tensor(y_dna_seqs, dtype=torch.float)

        return X_dna_seqs_onehot_padding, y_dna_seqs

    def evaluation_for_single_sequence(self, dna_seq: str):
        """Evaluation of input DNA sequence string, returns the predicted count using pretrained saved model.
        Args:
            dna_seq (str): DNA sequence string
        Returns:
            out(tensor): predicted CG count
        """
        seq = [[x for x in dnaseq_to_intseq(dna_seq)]]
        sample_x, sample_y = self.inference_for_single_sequence(seq, 128)
        dl = DataLoader(
            MyDataset(sample_x, sample_y), batch_size=1, collate_fn=PadSequence()
        )

        model = self.model
        model.eval()
        with torch.no_grad():
            for batch in dl:
                inp, length, target = batch
                out = model(inp, length)
                out = out.squeeze()
                return out.item()
