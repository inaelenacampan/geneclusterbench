import argparse
import pickle as pk
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

MODEL_NAME = "Rostlab/ProstT5"
DEFAULT_CACHE_DIR = "/hps/nobackup/jlees/campan/cache/huggingface"


def parse_args():
    """Read command-line options describing the input FASTA, output pickle,
    and execution resources for the ProstT5 embedding step."""
    parser = argparse.ArgumentParser(
        description="Embed protein sequences from a FASTA file with ProstT5."
    )
    parser.add_argument(
        "--input-fasta",
        required=True,
        type=Path,
        help="Amino-acid FASTA file to embed (e.g. *_for_clustering_aa.fasta).",
    )
    parser.add_argument(
        "--out-pk",
        type=Path,
        default=Path("./embeddings.pk"),
        help="Path to write the pickled {ids, prots, embd} dict to.",
    )
    parser.add_argument(
        "--nthreads",
        "-j",
        type=int,
        default=8,
        help="Number of CPU threads passed to torch.set_num_threads.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="HuggingFace cache directory for the ProstT5 tokenizer/model.",
    )
    parser.add_argument(
        "--max-prots",
        type=int,
        default=1_000_000,
        help="Stop reading the FASTA file after this many records.",
    )
    return parser.parse_args()


def read_fasta(path, max_prots):
    """Read a FASTA file into parallel lists of ids and sequences."""
    listofids = []
    listofprots = []
    protid = ""
    protseq = ""
    with path.open("r") as handle:
        for line in handle:
            if ">" in line:
                if protid != "":
                    listofprots.append(protseq.replace("\n", "").replace("*", ""))
                    listofids.append(protid)
                    protseq = ""
                protid = line.split(" ")[0].replace(">", "").strip()
            else:
                protseq += line
            if len(listofprots) > max_prots:
                break
    # flush the last record
    if protid != "" and (not listofprots or listofids[-1] != protid):
        listofprots.append(protseq.replace("\n", "").replace("*", ""))
        listofids.append(protid)
    return listofids, listofprots


def prepare_sequences(listofprots):
    """Replace rare/ambiguous amino acids with X, add whitespace between
    residues, and prepend the ProstT5 direction token (<AA2fold> for
    amino-acid sequences, <fold2AA> for lower-case 3Di sequences)."""
    listofprots = [
        " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in listofprots
    ]
    listofprots = [
        "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s
        for s in listofprots
    ]
    return listofprots


def load_model(cache_dir, device):

    print("before tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME,
        do_lower_case=False,
        legacy=True,
        cache_dir=cache_dir,
    )
    print("after tokenizer")

    print("before model")
    print("MODEL_NAME =", MODEL_NAME)
    print("cache_dir =", cache_dir)
    model = T5EncoderModel.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
    )
    print("after model")

    print("before to(device)")
    model = model.to(device)
    print("after to(device)")

    print("before precision")
    model.float() if device.type == "cpu" else model.half()
    print("after precision")

    model.eval()
    print("after eval")

    return tokenizer, model


def embed_sequences(listofprots, tokenizer, model, device):
    """Embed each prepared sequence individually and mean-pool over residue
    positions (excluding the prefix token and the trailing EOS token) to get
    one fixed-length vector per protein."""
    outembeds = []
    for i, iP in enumerate(listofprots):
        print(f"> Embedding protein {i + 1}/{len(listofprots)}")
        ids = tokenizer(
            [iP], add_special_tokens=True, padding="longest", return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            embedding_rpr = model(
                ids.input_ids, attention_mask=ids.attention_mask
            )

        # number of real (non-padded) tokens for this sequence, including the
        # prepended <AA2fold>/<fold2AA> token and the trailing </s> token
        seq_len = int(ids.attention_mask[0].sum().item())
        # drop position 0 (prefix token) and the final token (EOS) so we only
        # average over the actual amino-acid / 3Di residue positions
        outembeds.append(
            embedding_rpr.last_hidden_state[0, 1 : seq_len - 1].mean(dim=0).cpu().numpy()
        )
        del ids, embedding_rpr

    return np.vstack(outembeds)


def main():
    args = parse_args()
    run_start = time.time()

    print("> Initialising PyTorch")
    torch.set_num_threads(args.nthreads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"> Reading proteins from {args.input_fasta}")
    listofids, listofprots = read_fasta(args.input_fasta, args.max_prots)
    print(f"> Got {len(listofids)} proteins")

    print("> Loading model")
    tokenizer, model = load_model(args.cache_dir, device)
    print("here")
    prepared = prepare_sequences(listofprots)

    print("> Generating embeddings")
    matrixofembds = embed_sequences(prepared, tokenizer, model, device)

    outdict = {
        "ids": listofids,
        "prots": prepared,
        "embd": matrixofembds,
    }

    args.out_pk.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pk.open("wb") as f:
        pk.dump(outdict, f)

    print(f"> Wrote {matrixofembds.shape} embedding matrix to {args.out_pk}")
    print(f"> Done in {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
