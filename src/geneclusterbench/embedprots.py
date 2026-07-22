import argparse
import pickle as pk
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

MODEL_NAME = "Rostlab/ProstT5"
DEFAULT_CACHE_DIR = "/nfs/research/jlees/campan/cache/huggingface"


def parse_args():
    """Read command-line options describing the input FASTA file(s), output
    pickle path(s), and execution resources for the ProstT5 embedding step.

    Supports embedding many FASTA files in a single process (one model
    load, many inputs) instead of one process per file.
    """
    parser = argparse.ArgumentParser(
        description="Embed protein sequences from one or more FASTA files with ProstT5."
    )
    parser.add_argument(
        "--input-fasta",
        required=True,
        type=Path,
        nargs="+",
        help="One or more amino-acid FASTA files to embed "
        "(e.g. *_for_clustering_aa.fasta). The model is loaded once and "
        "reused for all of them.",
    )
    parser.add_argument(
        "--out-pk",
        type=Path,
        nargs="+",
        default=None,
        help="Output pickle path(s), one per --input-fasta, in the same "
        "order. If omitted, each output is written next to the matching "
        "--out-dir using the input file's stem.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write outputs into when --out-pk is not given "
        "explicitly. Output filenames are '<fasta stem>.pk'.",
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
        help="Stop reading each FASTA file after this many records.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of sequences embedded per forward pass. Increase on "
        "GPU with plenty of memory, decrease on CPU or if you hit OOM.",
    )

    args = parser.parse_args()

    if args.out_pk is not None and len(args.out_pk) != len(args.input_fasta):
        parser.error("--out-pk must be given once per --input-fasta")

    if args.out_pk is None:
        out_dir = args.out_dir or Path(".")
        args.out_pk = [out_dir / f"{fasta.stem}.pk" for fasta in args.input_fasta]

    return args


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
    print("> Loading tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME,
        do_lower_case=False,
        legacy=True,
        cache_dir=cache_dir,
    )

    print("> Loading model", MODEL_NAME, "from cache_dir =", cache_dir)
    model = T5EncoderModel.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
    )
    model = model.to(device)
    model.float() if device.type == "cpu" else model.half()
    model.eval()
    print("> Model ready")

    return tokenizer, model


def embed_sequences(listofprots, tokenizer, model, device, batch_size=32):
    """Embed sequences in batches and mean-pool over residue positions
    (excluding the prefix token and the trailing EOS token) to get one
    fixed-length vector per protein.

    Sequences are processed longest-first-sorted-by-length within batches
    to minimise padding waste, then results are returned in the original
    input order.
    """
    n = len(listofprots)
    results = [None] * n
    order = sorted(range(n), key=lambda i: len(listofprots[i]))

    for start in range(0, n, batch_size):
        batch_idx = order[start : start + batch_size]
        batch_seqs = [listofprots[i] for i in batch_idx]

        ids = tokenizer(
            batch_seqs, add_special_tokens=True, padding="longest", return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            embedding_rpr = model(ids.input_ids, attention_mask=ids.attention_mask)

        for j, idx in enumerate(batch_idx):
            # number of real (non-padded) tokens for this sequence, including
            # the prepended <AA2fold>/<fold2AA> token and the trailing </s>
            seq_len = int(ids.attention_mask[j].sum().item())
            # drop position 0 (prefix token) and the final token (EOS) so we
            # only average over the actual amino-acid / 3Di residue positions
            vec = (
                embedding_rpr.last_hidden_state[j, 1 : seq_len - 1]
                .mean(dim=0)
                .cpu()
                .numpy()
            )
            results[idx] = vec

        del ids, embedding_rpr

        done = min(start + batch_size, n)
        print(f"> Embedded {done}/{n}")

    return np.vstack(results)


def main():
    args = parse_args()
    run_start = time.time()

    print("> Initialising PyTorch")
    torch.set_num_threads(args.nthreads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("> Using device:", device)
    if device.type == "cpu":
        print(
            "! WARNING: running on CPU. This model is much faster on GPU; "
            "check your job submission requests a GPU if one is expected."
        )

    # Load the model exactly once, then reuse it for every input FASTA.
    tokenizer, model = load_model(args.cache_dir, device)

    for fasta_path, out_path in zip(args.input_fasta, args.out_pk):
        file_start = time.time()
        print(f"\n> Reading proteins from {fasta_path}")
        listofids, listofprots = read_fasta(fasta_path, args.max_prots)
        print(f"> Got {len(listofids)} proteins")

        prepared = prepare_sequences(listofprots)

        print(f"> Generating embeddings (batch_size={args.batch_size})")
        matrixofembds = embed_sequences(
            prepared, tokenizer, model, device, batch_size=args.batch_size
        )

        outdict = {
            "ids": listofids,
            "prots": prepared,
            "embd": matrixofembds,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pk.dump(outdict, f)

        print(f"> Wrote {matrixofembds.shape} embedding matrix to {out_path}")
        print(f"> {fasta_path.name} done in {time.time() - file_start:.1f}s")

    print(f"\n> All done in {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
