import string, re
import numpy as np
import torch

MAX_INPUT_SEQ = 500
ID_col = 'ID'
sequence_col = "Sequence"
metal_list = ["ZN", "CA", "MG", "MN"]
LMetalSite_threshold = [0.42, 0.34, 0.5, 0.47]

NN_config = {
    'feature_dim': 1024,
    'hidden_dim': 64,
    'num_encoder_layers': 2,
    'num_heads': 4,
    'augment_eps': 0.05,
    'dropout': 0.2,
}


class MetalDataset:
    def __init__(self, df, protein_features):
        self.df = df
        self.protein_features = protein_features
        self.feat_dim = NN_config['feature_dim']

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, idx):
        seq_id = self.df.loc[idx, ID_col]
        protein_feat = self.protein_features[seq_id]
        return protein_feat

    def padding(self, batch, maxlen):
        batch_protein_feat = []
        batch_protein_mask = []
        for protein_feat in batch:
            padded_protein_feat = np.zeros((maxlen, self.feat_dim))
            padded_protein_feat[:protein_feat.shape[0]] = protein_feat
            padded_protein_feat = torch.tensor(padded_protein_feat, dtype = torch.float)
            batch_protein_feat.append(padded_protein_feat)

            protein_mask = np.zeros(maxlen)
            protein_mask[:protein_feat.shape[0]] = 1
            protein_mask = torch.tensor(protein_mask, dtype = torch.long)
            batch_protein_mask.append(protein_mask)

        return torch.stack(batch_protein_feat), torch.stack(batch_protein_mask)

    def collate_fn(self, batch):
        maxlen = max([protein_feat.shape[0] for protein_feat in batch])
        batch_protein_feat, batch_protein_mask = self.padding(batch, maxlen)

        return batch_protein_feat, batch_protein_mask, maxlen


def process_fasta(fasta_file):
    ID_list = []
    seq_list = []

    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            name_item = line[1:-1].split("|")
            ID = "_".join(name_item[0:min(2, len(name_item))])
            ID = re.sub(" ", "_", ID)
            ID_list.append(ID)
        elif line[0] in string.ascii_letters:
            seq_list.append(line.strip().upper())

    if len(ID_list) == len(seq_list):
        if len(ID_list) > MAX_INPUT_SEQ:
            return 1
        else:
            return [ID_list, seq_list]
    else:
        return -1
