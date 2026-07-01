# Copyright Gerry Tonkin-Hill 2019
# import libraries

import sys, os
import argparse
from collections import OrderedDict, defaultdict
import gffutils
from Bio import SeqIO
from Bio import Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from io import StringIO
import numpy as np
import random
from dendropy.simulate import treesim
from dendropy.model import reconcile
from dendropy import TaxonNamespace
import copy
import math
from Bio.SeqFeature import SeqFeature, FeatureLocation
from BCBio import GFF
import matplotlib.pyplot as plt

# codon = sequence of three consecutive nucleotides (the building blocks of DNA and RNA) that acts as the basic unit of genetic information

codons = [
    'ATA', 'ATC', 'ATT', 'ATG', 'ACA', 'ACC', 'ACG', 'ACT', 'AAC', 'AAT',
    'AAA', 'AAG', 'AGC', 'AGT', 'AGA', 'AGG', 'CTA', 'CTC', 'CTG', 'CTT',
    'CCA', 'CCC', 'CCG', 'CCT', 'CAC', 'CAT', 'CAA', 'CAG', 'CGA', 'CGC',
    'CGG', 'CGT', 'GTA', 'GTC', 'GTG', 'GTT', 'GCA', 'GCC', 'GCG', 'GCT',
    'GAC', 'GAT', 'GAA', 'GAG', 'GGA', 'GGC', 'GGG', 'GGT', 'TCA', 'TCC',
    'TCG', 'TCT', 'TTC', 'TTT', 'TTA', 'TTG', 'TAC', 'TAT', 'TGC', 'TGT', 'TGG'
]

# seq -> string kind of object for bioinformatics

codons = [Seq(c) for c in codons]

# genetic codon translation table (codon → amino acid lookup)
# Start Codon: AUG (Methionine) is the universal start codon that initiates translation.
# Stop Codons: UAA, UAG, and UGA are stop codons that terminate translation.

translation_table = np.array([[[b'K', b'N', b'K', b'N', b'X'],
                               [b'T', b'T', b'T', b'T', b'T'],
                               [b'R', b'S', b'R', b'S', b'X'],
                               [b'I', b'I', b'M', b'I', b'X'],
                               [b'X', b'X', b'X', b'X', b'X']],
                              [[b'Q', b'H', b'Q', b'H', b'X'],
                               [b'P', b'P', b'P', b'P', b'P'],
                               [b'R', b'R', b'R', b'R', b'R'],
                               [b'L', b'L', b'L', b'L', b'L'],
                               [b'X', b'X', b'X', b'X', b'X']],
                              [[b'E', b'D', b'E', b'D', b'X'],
                               [b'A', b'A', b'A', b'A', b'A'],
                               [b'G', b'G', b'G', b'G', b'G'],
                               [b'V', b'V', b'V', b'V', b'V'],
                               [b'X', b'X', b'X', b'X', b'X']],
                              [[b'*', b'Y', b'*', b'Y', b'X'],
                               [b'S', b'S', b'S', b'S', b'S'],
                               [b'*', b'C', b'W', b'C', b'X'],
                               [b'L', b'F', b'L', b'F', b'X'],
                               [b'X', b'X', b'X', b'X', b'X']],
                              [[b'X', b'X', b'X', b'X', b'X'],
                               [b'X', b'X', b'X', b'X', b'X'],
                               [b'X', b'X', b'X', b'X', b'X'],
                               [b'X', b'X', b'X', b'X', b'X'],
                               [b'X', b'X', b'X', b'X', b'X']]])

# ASCII codes
# 0A 1C 2G 3T

reduce_array = np.full(200, 4)
reduce_array[[65, 97]]  = 0
reduce_array[[67, 99]]  = 1
reduce_array[[71, 103]] = 2
reduce_array[[84, 116]] = 3

absolute_gene_map = {}
absolute_gene_ind = 0

def translate(seq):
    # encode sequence
    indices = reduce_array[np.fromstring(seq, dtype=np.int8)]
    # Returns an array of amino acid bytes
    return translation_table[
        indices[np.arange(0, len(seq), 3)], indices[np.arange(1, len(seq), 3)],
        indices[np.arange(2, len(seq), 3)]].tostring().decode('ascii')


def get_codon(index, strand="+"):
    codon = codons[index]
    if strand == "-":
        codon = codon.reverse_complement()
    return np.array(list(str(codon)))


def clean_gff_string(gff_string):
    # list of strings
    splitlines = gff_string.splitlines()
    lines_to_delete = []

    for index in range(len(splitlines)):
        # directive lines from a GFF
        if '##sequence-region' in splitlines[index]:
            lines_to_delete.append(index)

    for index in sorted(lines_to_delete, reverse=True):
        # delete object
        del splitlines[index]
    cleaned_gff = "\n".join(splitlines)
    return cleaned_gff


def reverse_complement_text(seq):
    return str(Seq(str(seq)).reverse_complement())


def validate_no_internal_stop_codons(gene_sequence, entry):
    translated = str(gene_sequence.translate())[:-1]  # drop trailing stop
    stop_positions = [i + 1 for i, aa in enumerate(translated) if aa == "*"]
    if stop_positions:
        raise RuntimeError(
            "Internal stop codon(s) found in CDS "
            f"{entry.id} ({entry.seqid}:{entry.start}-{entry.stop}, "
            f"strand {entry.strand}) at codon(s) {stop_positions}"
        )
    return 0


def normalise_deletion_indices(indices):
    return np.unique(indices.astype(int))


def cds_intervals_overlap(left_entry, right_entry):
    if left_entry.seqid != right_entry.seqid:
        return False
    return left_entry.start <= right_entry.stop and right_entry.start <= left_entry.stop


def overlaps_any_fixed_core(entry, fixed_core_entries):
    return any(cds_intervals_overlap(entry, fixed_entry) for fixed_entry in fixed_core_entries)


def simulate_img_with_mutation(in_tree,
                               gain_rate,
                               loss_rate,
                               mutation_rate,
                               random_state,
                               ngenes    = 100,
                               min_ncore = 10,
                               max_ncore = 99999999):
    # simulate accessory p/a using infintely many genes model -> see paper of Baumdicker
    n_additions = 0

    # in_tree : phylogenetic tree
    # gain rate ->  a gene gain is the first occurence of a new gene in a population 

    # genes are lost with exponential probability, and new genes are gained via a Poisson process

    for node in in_tree.preorder_node_iter():
        node.acc_genes = []
        if node.parent_node is not None:
            # probability to keep a gene
            p_keep = np.exp(-(node.edge.length * loss_rate / 2.0))

            # each gene inherited from the parent survives independently with a certain probability

            to_inherit = [
                g for g in node.parent_node.acc_genes
                if np.random.random() < p_keep
            ]

            # simulate new genes with lengths sampled uniformly.
            # poisson distribution

            n_new = np.random.poisson(lam=node.edge.length * gain_rate / 2.0)
            lengths = np.random.uniform(low=0.0,
                                        high=node.edge.length,
                                        size=n_new)
            for l in lengths:
                # simulate loss using this length
                if np.random.poisson(lam=l * loss_rate / 2.0) > 0:
                    n_new -= 1

            # add new genes to node
            node.acc_genes = to_inherit + list(
                range(n_additions, n_additions + n_new))
            n_additions += n_new

    print("\t- Accessory size: ", n_additions)
    # Now add core. Gene IDs must stay within the sampled source genes.
    remaining_gene_slots = ngenes - n_additions
    if remaining_gene_slots < 0:
        raise RuntimeError(
            f"Simulated {n_additions} accessory genes, but only {ngenes} "
            "source genes were sampled. Increase --n_sim_genes or lower "
            "--gain_rate/--pop_size."
        )

    ncore = min(remaining_gene_slots, max_ncore)
    if ncore < min_ncore:
        print(
            "\t- Core size capped by sampled source genes: ",
            ncore,
            f"(requested minimum {min_ncore})",
        )

    core_genes = list(range(n_additions, n_additions + ncore))
    for node in in_tree.preorder_node_iter():
        node.acc_genes += core_genes

    # Now add mutations
    for node in in_tree.preorder_node_iter():
        node.gene_mutations = defaultdict(list)
        if node.parent_node is not None:
            # copy mutations from parent
            for g in node.acc_genes:
                if g in node.parent_node.gene_mutations:
                    node.gene_mutations[g] = node.parent_node.gene_mutations[
                        g].copy()
            # add mutations
            for g in node.acc_genes:
                n_new = np.random.poisson(lam=node.edge.length *
                                          mutation_rate / 2.0,
                                          size=1)[0]
                # get random location
                locations = list(np.random.uniform(low=0.0, high=1,
                                                   size=n_new))
                # associated codon
                mutations = [(random_state.sample(range(0, len(codons)), 1)[0], l)
                             for l in locations]
                node.gene_mutations[g] += mutations

    return in_tree


def simulate_pangenome(ngenes, nisolates, effective_pop_size, gain_rate,
                       loss_rate, mutation_rate, max_core, random_state):

    # simulate a phylogeny using the coalscent
    
    sim_tree = treesim.pure_kingman_tree(
        # TaxonNamespace : manage unique identities of operational taxonomic units (OTUs) across multiple data structures
        taxon_namespace = TaxonNamespace([str(i) for i in range(1, 1 + nisolates)]),
        pop_size        = effective_pop_size,
        rng             = random_state,
    )

    basic_tree = copy.deepcopy(sim_tree)

    # simulate gene p/a and mutation
    # using infintely many genes model 
    sim_tree = simulate_img_with_mutation(sim_tree,
                                          gain_rate     = gain_rate,
                                          loss_rate     = loss_rate,
                                          mutation_rate = mutation_rate,
                                          ngenes        = ngenes,
                                          max_ncore     = max_core,
                                          random_state  = random_state,)

    # get genes and mutations for each isolate
    gene_mutations = []
    for leaf in sim_tree.leaf_node_iter():
        gene_mutations.append([[g, leaf.gene_mutations[g]]
                               for g in leaf.acc_genes])

    return (gene_mutations, basic_tree)


def get_gene_id(seq):
    theid = 0
    global absolute_gene_ind, absolute_gene_map;
    if seq in absolute_gene_map:
        theid = absolute_gene_map[seq]
    else:
        theid = absolute_gene_ind
        absolute_gene_map[seq] = absolute_gene_ind
        absolute_gene_ind += 1

    return theid

def draw_phylogenetic_tree(filepath, folder, gain_rate, loss_rate, mutation_rate):
    # treefile needs to be a .nwk format
    tree = Phylo.read(filepath, "newick")

    fig, ax = plt.subplots(figsize=(12, 8))
    Phylo.draw(tree, axes=ax)
    file_name_tree = ("sim_gr_" + str(gain_rate) + "_lr_" +
              str(loss_rate) + "_mu_" + str(mutation_rate) + "tree.png")
    plt.savefig(f"{folder}/{file_name_tree}", dpi=300)
    return

# Main function
def add_diversity(gfffile, nisolates, effective_pop_size, gain_rate, loss_rate,
                  mutation_rate, n_sim_genes, prefix, max_core, random_state):

    # nisolates : number of simulation to produce

    # reading sequences
    print("> Opening GFF3 file")
    with open(gfffile, 'r') as infile:
        lines = infile.read().replace(',','')

    split = lines.split('##FASTA')
    if len(split) != 2:
        print("Problem reading GFF3 file: ", gfffile)
        raise RuntimeError("Error reading GFF3 input!")

    with StringIO(split[1]) as temp_fasta:
        sequences = list(SeqIO.parse(temp_fasta, 'fasta'))

    print("> Sequences read")
    seq_dict = OrderedDict()
    for seq in sequences:
        seq_dict[seq.id] = np.array(list(str(seq.seq)))

    gene_seq_dict = {}
    original_gene_ids = {}
    # geneid_* is protein-based; keep exact nucleotide sequences separately.
    original_gene_sequences = {}

    parsed_gff = gffutils.create_db(
        clean_gff_string(split[0]),
        dbfn             = ":memory:",
        force            = True,
        keep_order       = False,
        merge_strategy   = "create_unique",
        sort_attribute_values = True,
        from_string      = True,
    )

    # Get gene entries to modify
    all_gene_locations  = []
    gene_locations      = []
    prev_end_by_seqid   = defaultdict(lambda: -1)
    gene_seqs           = []

    # CD = coding sequence

    print("> Iterating over CDS entries...")
    for entry in parsed_gff.all_features(featuretype=()):
        if "CDS" not in entry.featuretype: continue

        left  = entry.start - 1
        right = entry.stop

        # Extract the nucleotide sequence
        gene_sequence = Seq(''.join(seq_dict[entry.seqid][left:right]))

        # take the reverse
        if entry.strand == "-":
            gene_sequence = gene_sequence.reverse_complement()

        validate_no_internal_stop_codons(gene_sequence, entry)

        gene_seq_to_save = copy.deepcopy(gene_sequence)
        # print(gene_seq_to_save)

        # translate to amino acid
        # possible bug for the annotation of the codon?

        gene_sequence = gene_sequence.translate(stop_symbol = "")

        # print(gene_sequence); sys.exit()

        geneid = get_gene_id(gene_sequence)         # The IDs must be of the translated genes, i.e. of the AA sequences. That is what makes sense.
        # geneid = get_gene_id(gene_seq_to_save)
        gene_id = "geneid_" + str(geneid)
        original_gene_ids[entry.id] = gene_id
        original_gene_sequences[entry.id] = str(gene_seq_to_save)
        gene_seq_dict.setdefault(gene_id, str(gene_seq_to_save))

        gene_seqs.append(SeqRecord(gene_sequence, id=entry.id, description=gene_id))

        all_gene_locations.append(entry)
        prev_end = prev_end_by_seqid[entry.seqid]
        if entry.start <= prev_end:
            prev_end_by_seqid[entry.seqid] = max(prev_end, entry.end)
            continue
        prev_end_by_seqid[entry.seqid] = entry.end
        gene_locations.append(entry)
    print("> Done!")


    # print(seq_dict)
    # print(gene_locations); sys.exit()

    # Check that all coordinates of genes are effectively inside the contigs

    # for iG in gene_locations:
    #     # print(dir(iG))
    #     # print(iG.source, iG.start, iG.stop, iG.strand, iG.seqid)
    #     # print(seq_dict[iG.seqid])
    #     theseql = len(seq_dict[iG.seqid])
    #     # if iG.start > theseql or iG.stop >  theseql:
    #     #     print(iG.seqid, iG.start, iG.stop, theseql)
    #     print(iG.seqid, iG.start, iG.stop, theseql)
    # sys.exit()

    # sub-sample genes so that some are conserved
    if n_sim_genes < 1:
        raise RuntimeError("--n_sim_genes must be at least 1")
    if n_sim_genes > len(gene_locations):
        raise RuntimeError(
            f"Requested --n_sim_genes={n_sim_genes}, but only "
            f"{len(gene_locations)} non-overlapping CDS entries are available"
        )
    print("> Subsampling genes...")
    sampled_gene_locations = []
    available_gene_locations = gene_locations.copy()

    while len(sampled_gene_locations) < n_sim_genes:
        if not available_gene_locations:
            raise RuntimeError(
                f"Could only sample {len(sampled_gene_locations)} CDS entries "
                f"without fixed-core overlaps, but --n_sim_genes={n_sim_genes}"
            )

        candidate = random_state.sample(available_gene_locations, 1)[0]
        available_gene_locations.remove(candidate)

        tentative_sample = sampled_gene_locations + [candidate]
        tentative_sample_ids = {entry.id for entry in tentative_sample}
        fixed_core_entries = [
            entry for entry in all_gene_locations
            if entry.id not in tentative_sample_ids
        ]

        if overlaps_any_fixed_core(candidate, fixed_core_entries):
            print(
                "WARNING: sampled CDS overlaps fixed core CDS; resampling:",
                candidate.id,
            )
            continue

        sampled_gene_locations.append(candidate)

    gene_locations = sampled_gene_locations
    sampled_gene_ids = {entry.id for entry in gene_locations}
    print("> Done!")

    print("> Simulating presence/absence matrix and gene mutations...")
    # simulate presence/absence matrix and gene mutations (only swap codons)
    pan_sim, sim_tree = simulate_pangenome(
        ngenes        = len(gene_locations),
        nisolates     = nisolates,
        effective_pop_size = effective_pop_size,
        gain_rate     = gain_rate,
        loss_rate     = loss_rate,
        mutation_rate = mutation_rate,
        max_core      = max_core,
        random_state  = random_state,
    )
    max_gene_index = max((gene[0] for pan in pan_sim for gene in pan), default=-1)
    if max_gene_index >= len(gene_locations):
        raise RuntimeError(
            f"Simulation generated gene index {max_gene_index}, but only "
            f"{len(gene_locations)} sampled CDS entries are available"
        )

    print("> Done!")

    # write out tree
    print("> Writing out phylogenetic tree in Newick format...")
    sim_tree.write(path = prefix + "_sim_tree.nwk", schema = "newick")
    print("> Done!")

    #Modify each gene
    print("> Modifying all genes from the simulated pangenome...")

    # print((pan_sim[0][4])); sys.exit()

    for i, pan in enumerate(pan_sim): # Iterate over samples/isolates/assemblies
        print("\n\t- Modifying simulated genome", i)
        temp_seq_dict = copy.deepcopy(seq_dict)
        included_genes = set()
        n_mutations = 0
        # Feature IDs are unique in the output GFF; use them for exact NT validation.
        feature_seq_dict = {}
        # print("here1")
        for gene in pan: # Iterate over mutated genes, and mutate them
            # gene is a list with first element an index and the second a list of duples (integer, float)
            # print(gene); sys.exit()
            # if gene[0] not in range(len(gene_locations)): continue
            entry = gene_locations[gene[0]]
            included_genes.add(gene[0])

            left  = entry.start - 1
            right = entry.stop

            if right < left: raise RuntimeError("Error issue with left/right!")

            start_sites = list(range(left, right, 3))[1:-1]

            if gene[1] and not start_sites:
                raise RuntimeError(
                    "Cannot apply internal codon mutations to CDS "
                    f"{entry.id} ({entry.seqid}:{entry.start}-{entry.stop}, "
                    f"strand {entry.strand}) because it has no mutable "
                    "internal codon positions"
                )

            n_mutations += len(gene[1])

            # swap codons at chosen start sites
            for mutation in gene[1]:
                # find start site of codon swap
                start = start_sites[math.floor(mutation[1] * len(start_sites))]
                cod   = get_codon(index = mutation[0], strand = entry.strand)
                if (start < left) or ((start + 3) > (right)):
                    raise RuntimeError("Error issue with start!")
                temp_seq_dict[entry.seqid][start:(start + 3)] = cod

        if not included_genes:
            print(
                "WARNING: simulated genome",
                i,
                "has no sampled CDS entries retained; output GFF will contain only fixed core CDS entries",
            )

        # print("here2")

        # remove genes not in the accessory
        deleted_genes = 0
        GFF_entries = {}
        expected_seq_by_g = {}   # unambiguous NT sequence per gene-instance, keyed by loop index g
        d_index = defaultdict(lambda: np.array([])) # Here we store the indices that indicate what genes to remove
        for g, entry in enumerate(gene_locations):
            left = entry.start - 1
            right = entry.stop
            if right < left: raise RuntimeError("Error issue with left/right!")
            if g not in included_genes:
                deleted_genes += 1
                d_index[entry.seqid] = np.append(d_index[entry.seqid],
                                                 np.arange(left, right))

            gene_sequence = Seq(''.join(
                temp_seq_dict[entry.seqid][left:right]))
            if entry.strand == "-":
                gene_sequence = gene_sequence.reverse_complement()

            gene_seq_to_save = copy.deepcopy(gene_sequence)
            expected_seq_by_g[g] = str(gene_seq_to_save)   # keyed by g, no collisions possible
            # print(gene_seq_to_save)
            # possible bug for the annotation of the codon?
            gene_sequence = gene_sequence.translate(stop_symbol = "")
            geneid = get_gene_id(gene_sequence)         # The IDs must be of the translated genes, i.e. of the AA sequences. That is what makes sense.
            # geneid = get_gene_id(gene_seq_to_save)
            gene_id = "geneid_" + str(geneid)
            gene_seq_dict.setdefault(gene_id, str(gene_seq_to_save))
            gene_seqs.append(
                SeqRecord(gene_sequence, id = entry.id, description = gene_id)
            )
            if g in included_genes:
                if not entry.seqid in GFF_entries:
                    GFF_entries[entry.seqid] = []

                GFF_entries[entry.seqid].append(copy.deepcopy(entry))
                # GFF_entries[entry.seqid][-1].id = entry.id + " geneid_" + str(geneid)
                feature_id = entry.id + "-" + gene_id # Don't add spaces, some software might not expect them (though they are allowed in GFF3 in principle...)
                GFF_entries[entry.seqid][-1].id = feature_id
                feature_seq_dict[feature_id] = str(gene_seq_to_save)


        for entry in all_gene_locations:
            if entry.id in sampled_gene_ids:
                continue

            if entry.seqid not in GFF_entries:
                GFF_entries[entry.seqid] = []

            GFF_entries[entry.seqid].append(copy.deepcopy(entry))
            feature_id = entry.id + "-" + original_gene_ids[entry.id]
            GFF_entries[entry.seqid][-1].id = feature_id
            feature_seq_dict[feature_id] = original_gene_sequences[entry.id]


        # print("here3")

        for entryid in d_index:
            # print("\n", entry.seqid, "\n", temp_seq_dict[entryid].shape, "\n", d_index[entryid])
            raw_deleted = d_index[entryid].astype(int)
            deleted = normalise_deletion_indices(d_index[entryid])

            n_dup = len(raw_deleted) - len(deleted)
            if n_dup:
                print(entryid, "duplicated deleted indices:", n_dup)

            if entryid in GFF_entries:
                retained_entries = []
                for gene in GFF_entries[entryid]:
                    gene_left = gene.start - 1
                    gene_right = gene.stop
                    overlaps_deleted = bool(np.any((deleted >= gene_left) & (deleted < gene_right)))
                    if overlaps_deleted:
                        print(entryid, "dropped CDS overlapping deleted sequence:", gene.id)
                        continue

                    tmpsum = int(np.sum(deleted < gene_left))
                    if tmpsum:
                        gene.start -= tmpsum
                        gene.stop -= tmpsum
                    retained_entries.append(gene)

                GFF_entries[entryid] = retained_entries


            temp_seq_dict[entryid] = np.delete(temp_seq_dict[entryid], deleted)


        record_list = []
        for iS in temp_seq_dict: # These are the contigs
            record_list.append(SeqRecord(Seq(''.join(temp_seq_dict[iS])), id = iS, description = ""))
            record_list[-1].features = []
            if iS in GFF_entries:
                GFF_entries[iS].sort(key=lambda gene: (gene.start, gene.stop, gene.id))
                for iG in GFF_entries[iS]: # And these, the "features" (i.e. the genes)
                    qualifiers = {
                        "source"    : "simulation",
                        "ID"        : iG.id,
                        "score"     : 1.0,
                    }
                    feature = SeqFeature(
                        # FeatureLocation(iG.start    if iG.strand == "+" else iG.start + 3, # Start
                        FeatureLocation(iG.start - 1 if iG.strand == "+" else iG.start + 2, # Start
                                        # iG.stop - 3 if iG.strand == "+" else iG.stop, # End
                                        # iG.stop - 2 if iG.strand == "+" else iG.stop + 1, # End
                                        iG.stop - 3 if iG.strand == "+" else iG.stop + 0, # End

                                        strand = 1   if iG.strand == "+" else -1), # strand
                        type = "CDS",
                        qualifiers = qualifiers,
                    )

                    # print(iG.start, iG.stop)
                    # if iG.strand == "+":
                    #     # print(iG.start - 1, iG.stop - 2, iG.stop - 2 - (iG.start - 1) + 1, (iG.stop - 2 - (iG.start - 1) + 1)/3)
                    #     print(iG.start, iG.stop - 3, iG.stop - 3 - (iG.start) + 1, (iG.stop - 3 - (iG.start) + 1)/3)
                    # else:
                    #     # print(iG.start + 3, iG.stop + 1, iG.stop + 1 - (iG.start + 3) + 1, (iG.stop + 1 - (iG.start + 3) + 1)/3)
                    #     print(iG.start + 3, iG.stop, iG.stop - (iG.start + 3) + 1, (iG.stop - (iG.start + 3) + 1)/3)


                    # sys.exit()
                    # if iG.stop > len(record_list[-1].seq):
                    #     print(len(record_list[-1].seq), iG.start, iG.stop)
                    #     print("\t========= HEY!!!!!")

                    contig_gene_seq = str(record_list[-1].seq[iG.start - 1 : iG.stop])
                    expected_gene_seq = feature_seq_dict[iG.id]
                    if iG.strand == "+":
                        # if record_list[-1].seq[iG.start - 1 : iG.stop] != gene_seq_dict[iG.id.split(" ")[1]]:
                        # if record_list[-1].seq[iG.start - 1 : iG.stop - 3] != gene_seq_dict[iG.id.split(" ")[1]][:-3]:
                        if contig_gene_seq[2:-3] != expected_gene_seq[2:-3]:
                            raise RuntimeError(f"CDS sequence mismatch for {iG.id} ({iG.seqid}:{iG.start}-{iG.stop}, strand {iG.strand})\nSeq from the contig:\n" + contig_gene_seq + f"\n> Expected CDS sequence:\n" + expected_gene_seq)
                    else:
                        # if "".join(["A" if el == "T" else "T" if el == "A" else "G" if el == "C" else "C" for el in record_list[-1].seq[iG.start - 1 : iG.stop][::-1] ]) != gene_seq_dict[iG.id.split(" ")[1]]:
                        # if "".join(["A" if el == "T" else "T" if el == "A" else "G" if el == "C" else "C" for el in record_list[-1].seq[iG.start - 1 : iG.stop][::-1] ])[:-3] != gene_seq_dict[iG.id.split(" ")[1]][:-3]:
                        contig_gene_rc = reverse_complement_text(contig_gene_seq)
                        if contig_gene_rc[2:-3] != expected_gene_seq[2:-3]:
                            raise RuntimeError(f"CDS sequence mismatch for {iG.id} ({iG.seqid}:{iG.start}-{iG.stop}, strand {iG.strand})\nSeq from the contig:\n" + contig_gene_seq + f"\n> Expected CDS sequence:\n" + expected_gene_seq + f"\n> Seq. reversed-complement:\n" + contig_gene_rc)

                    record_list[-1].features.append(feature)

        written_cds = sum(len(record.features) for record in record_list)
        if written_cds == 0:
            raise RuntimeError("No CDS annotations were written to the simulated GFF")
        print("# CDS written: ", written_cds)
        print("# Mutations in genome: ", n_mutations)
        print("# Genes deleted: ",       deleted_genes)

        # write out sequences
        print("# Writing sequences...")
        out_name = prefix + "_iso_" + str(i) + ".fasta"
        outfile = open(out_name, 'w')

        sequences = [
            SeqRecord(Seq(''.join(temp_seq_dict[s])), id = s, description = "")
            for s in temp_seq_dict
        ]

        SeqIO.write(sequences, outfile, 'fasta')
        # close file
        outfile.close()
        print("# Done!")
        print("# Writing GFF files per simulated assembly...")
        with open(out_name.replace(".fasta", ".gff"), "w") as f:
            GFF.write(record_list, f, include_fasta = True)
        print("# Done!")

    print("> Loop done!")

    # Write stupid tsv file without headers and with all the gffs and another one for all the fastas. Some programs require the former, the latter just in case
    outtxt = ""
    for i in range(nisolates):
        outtxt += prefix.split("/")[-1] + "_iso_" + str(i) + "\t" + prefix + "_iso_" + str(i) + ".fasta\n"
    with open(prefix + "_fasta_file.tsv", "w") as handle:
        handle.write(outtxt)

    outtxt = ""
    for i in range(nisolates):
        outtxt += prefix.split("/")[-1] + "_iso_" + str(i) + "\t" + prefix + "_iso_" + str(i) + ".gff\n"
    with open(prefix + "_gff_file.tsv", "w") as handle:
        handle.write(outtxt)

    # Because of panX, also transform each gff into a genbank annotation file


    # write out database for prokka
    print("> Writing database for Prokka...")
    prokka_db_name = prefix + "_prokka_DB.fasta"
    with open(prokka_db_name, 'w') as dboutfile:
        SeqIO.write(gene_seqs, dboutfile, 'fasta')
    print("> Done!")


    print("> Writing FASTA file for cluster algorithms with aminoacids and nucleotides...")
    cluster_name = prefix + "_for_clustering.fasta"
    cluster_name_aa = prefix + "_for_clustering_aa.fasta"
    gene_seqs_clustering = []
    gene_seqs_clustering_aa = []
    totalgeneset = set()
    outtxt = "gene_id\toriginal_gene\n"
    for iG in gene_seqs:
        if iG.description not in totalgeneset:
            gene_seqs_clustering_aa.append(
                SeqRecord(iG.seq, id = iG.description, description = "")
            )
            gene_seqs_clustering.append(
                SeqRecord(gene_seq_dict[iG.description], id = iG.description, description = "")
            )
            totalgeneset.add(iG.description)


            outtxt += iG.description + "\t" + iG.id + "\n"


    with open(cluster_name_aa, 'w') as clusteroutfile:
        SeqIO.write(gene_seqs_clustering_aa, clusteroutfile, 'fasta')

    with open(cluster_name, 'w') as clusteroutfile:
        SeqIO.write(gene_seqs_clustering, clusteroutfile, 'fasta')
    print("> Done!")

    print("> Writing truth matrix")
    truth_matrix_name = prefix + "_truth_matrix.tsv"
    with open(truth_matrix_name, 'w') as truthmatrixoutfile:
        truthmatrixoutfile.write(outtxt)
    print("> Done!")


    # write presence/absence file
    print("> Writing presence/absence file...")
    pa_by_iso = []
    for i, pan in enumerate(pan_sim):
        pa = set()
        for gene in pan:
            pa.add(gene[0])
        pa_by_iso.append(pa)

    out_name = prefix + "_presence_absence.csv"

    seen = set()
    with open(out_name, 'w') as outfile:
        outfile.write("\t".join(
            ["Gene"] + ["iso" + str(i)
                        for i in range(1, nisolates + 1)]) + "\n")
        for g, entry in enumerate(gene_locations):
            seen.add(entry.id)
            outfile.write("\t".join(
                [entry.id] +
                ["1" if g in pa_by_iso[i] else "0"
                 for i in range(nisolates)]) + "\n")

        for g, entry in enumerate(all_gene_locations):
            if entry.id in seen: continue
            outfile.write("\t".join([entry.id] +
                                    ["1" for i in range(nisolates)]) + "\n")

    print("> Done!")
    return


###########################################################################################
def main():

    parser = argparse.ArgumentParser(description=(
        'Simulates a pangenome using the infinitely many genes ' +
        'model and adds mutational variation to genes. Takes a gff3 file as input.'
    ))

    parser.add_argument('-g',
                        '--gff',
                        dest='gff',
                        type=str,
                        required=True,
                        help='Input gff file name')

    parser.add_argument('--nisolates',
                        dest='nisolates',
                        type=int,
                        default=100,
                        help='Number of genomes to simulate'
                             'Default = 100')

    parser.add_argument('--mutation_rate',
                        dest='mutation_rate',
                        type=float,
                        default=1e-14,
                        help='Mutation rate of genes.'
                             'Default = 1e-14')

    parser.add_argument('--gain_rate',
                        dest='gain_rate',
                        type=float,
                        default=1e-12,
                        help='Gain rate of accessory genes.'
                             'Default = 1e-12')

    parser.add_argument('--loss_rate',
                        dest='loss_rate',
                        type=float,
                        default=1e-12,
                        help='Loss rate of accessory genes.'
                             'Default = 1e-12')

    parser.add_argument('--pop_size',
                        dest='pop_size',
                        type=float,
                        default=10e6,
                        help='Effective population size. '
                             'Default = 10e6')

    parser.add_argument(
        '--n_sim_genes',
        dest='n_sim_genes',
        type=int,
        default=1000,
        help=('max number of genes that may be '
              'affected by the simulation. The rest will be left as is.'
              'Default = 1000'))

    parser.add_argument('--max_core',
                        dest='max_core',
                        type=int,
                        default=99999999,
                        help=('max number of core genes' +
                              'default=n_sim-accessory'))

    parser.add_argument('-o',
                        '--out',
                        dest='output_dir',
                        type=str,
                        required=True,
                        help='output directory')

    parser.add_argument('-s',
                        '--seed',
                        dest = 'seed',
                        type = int,
                        default  = 34,
                        required = False,
                        help = 'Seed for the random number generators')

    args = parser.parse_args()

    args.pop_size   = math.floor(args.pop_size)
    args.output_dir = os.path.join(args.output_dir, "")

    prefix = (args.output_dir + "sim_gr_" + str(args.gain_rate) + "_lr_" +
              str(args.loss_rate) + "_mu_" + str(args.mutation_rate))

    # adjust rates for popsize
    args.gain_rate      = 2.0 * args.pop_size * args.gain_rate
    args.loss_rate      = 2.0 * args.pop_size * args.loss_rate
    args.mutation_rate  = 2.0 * args.pop_size * args.mutation_rate

    # Fix random seed and make it deterministic

    np.random.seed(args.seed)
    rstate = random.Random(args.seed)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    print("> Starting to simulate")
    add_diversity(gfffile           = args.gff,
                  nisolates         = args.nisolates,
                  effective_pop_size= args.pop_size,
                  gain_rate         = args.gain_rate,
                  loss_rate         = args.loss_rate,
                  mutation_rate     = args.mutation_rate,
                  n_sim_genes       = args.n_sim_genes,
                  prefix            = prefix,
                  max_core          = args.max_core,
                  random_state      = rstate,
    )

    draw_phylogenetic_tree(f"{prefix}_sim_tree.nwk", args.output_dir, args.gain_rate, args.loss_rate, args.mutation_rate)
    print("> Simulation finished!")
    return


if __name__ == '__main__':
    main()
