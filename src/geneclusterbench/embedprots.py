import os, sys, re
from transformers import T5Tokenizer, T5EncoderModel
import torch
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import pandas as pd
import pickle as pk

#####################################################################################
nthreads = 8

print("> Initialising PyTorch")

torch.set_num_threads(nthreads)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print("> Loading models")
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False, legacy = True, cache_dir = "/hps/nobackup/jlees/vrbouza/cache/huggingface")

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/ProstT5", cache_dir = "/hps/nobackup/jlees/vrbouza/cache/huggingface").to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.float() if device.type=='cpu' else model.half()
# model.half()

# prepare your protein sequences/structures as a list.
# Amino acid sequences are expected to be upper-case ("PRTEINO" below)
# while 3Di-sequences need to be lower-case ("strctr" below).
# listofprots = ["PRTEINO", "strct"]
listofprots = []
listofids   = []
#### Let's start first with only one file

print("> Reading proteins")
thefile = "/nfs/research/jlees/vrbouza/data/2024_12_10_AtBSamDataset/mmseqs2_batch_13.faa"
# maxprots = 5000
#maxprots = 30000
maxprots = 1000000
with open(thefile, "r") as f:
    protid  = ""
    protseq = ""
    for line in f:
        if ">" in line:
            # New prot!
            if protid != "":
                listofprots.append(protseq.replace("\n", "").replace("*", ""))
                listofids.append(protid)
                protseq = ""

            protid = line.split(" ")[0].replace(">", "")
        else:
            protseq += line
        if len(listofprots) > maxprots: break

# print(listofprots)

# replace all rare/ambiguous amino acids by X (3Di sequences do not have those) and introduce white-space between all sequences (AAs and 3Di)
listofprots = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in listofprots]

# print(listofprots)
# The direction of the translation is indicated by two special tokens:
# if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
# if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
listofprots = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                      for s in listofprots
                    ]


#### All list processing
# listofprots = [listofprots[0]]
# print("> Tokenising proteins")
# # tokenize sequences and pad up to the longest sequence in the batch
# ids = tokenizer.batch_encode_plus(listofprots,
#                                   add_special_tokens = True,
#                                   padding = "longest",
#                                   return_tensors = 'pt').to(device)
#
# print("> Generating embeddings")
# # generate embeddings
#
# print(ids)
# print(ids.input_ids)
# # sys.exit()
# with torch.no_grad():
#     embedding_rpr = model(
#               ids.input_ids,
#               attention_mask=ids.attention_mask
#               )
# print(embedding_rpr)
# sys.exit()
# # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens, incl. prefix ([0,1:8])
# emb_0 = embedding_rpr.last_hidden_state[0, :] # shape (7 x 1024)
# # same for the second ([1,:]) sequence but taking into account different sequence lengths ([1,:6])
# emb_1 = embedding_rpr.last_hidden_state[1, 1:6] # shape (5 x 1024)
#
# # if you want to derive a single representation (per-protein embedding) for the whole protein
# emb_0_per_protein = emb_0.mean(dim = 0) # shape (1024)
#
# print(emb_0, emb_0.shape)
# print(emb_1, emb_1.shape)
# print(emb_0_per_protein, emb_0_per_protein.shape)

# listofembds = [embedding_rpr.last_hidden_state[i, 1:6].mean(dim = 0).cpu().numpy() for i in range(len(listofprots))]
# print(listofembds)
# matrixofembds = np.vstack(listofembds)

#### Individual protein processing
# listofprots = [listofprots[0]]
outembeds = []
for iP in listofprots:
    print("> Tokenising protein")
    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus([iP],
                                      add_special_tokens = True,
                                      padding = "longest",
                                      return_tensors = 'pt').to(device)

    print("> Generating embedding")
    # generate embeddings

    # print(ids)
    # print(ids.input_ids)
    # sys.exit()
    with torch.no_grad():
        embedding_rpr = model(
                  ids.input_ids,
                  attention_mask=ids.attention_mask
                  )
    # print(embedding_rpr)
    # sys.exit()

    outembeds.append(embedding_rpr.last_hidden_state[0, 1:6].mean(dim = 0).cpu().numpy())
    del ids,embedding_rpr


matrixofembds = np.vstack(outembeds)
outdict = {
    "ids"   : listofids,
    "prots" : listofprots,
    "embd"  : matrixofembds,
}

with open("./embeddings.pk", "wb") as f:
    pk.dump(outdict, f)
