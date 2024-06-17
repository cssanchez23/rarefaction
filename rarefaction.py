import argparse
import numpy as np
import pandas as pd
from numpy.random import RandomState
from pathlib import Path

"""THis code is created to rareify a OTU table or ASV table."""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Rarefy the OTU table to 500000 (wgs) or 10000 (amplicon) reads'
    )
    parser.add_argument(
        '-s', '--seq_type', metavar='seq_type', type=str, required=True,
        help='Sequencing type (wgs or amplicon)'
    )
    parser.add_argument(
        '-i', '--input_table', metavar='input_table', type=str, required=True,
        help='Input OTU or ASV table in csv format.'
    )
    parser.add_argument(
        '-o', '--output', metavar='output', type=str, required=True,
        help='Output rarefied OTU or ASV table in csv format'
    )
    return parser.parse_args()

def rarefaction(M, depth, seed=0):
    prng = RandomState(seed)  # reproducible results
    noccur = np.sum(M, axis=1)  # number of occurrences for each sample
    nvar = M.shape[1]  # number of variables

    Mrarefied = np.empty_like(M)
    for i in range(M.shape[0]):  # for each sample
        p = M[i] / float(noccur[i])  # relative frequency / probability
        choice = prng.choice(nvar, depth, p=p)
        Mrarefied[i] = np.bincount(choice, minlength=nvar)

    return Mrarefied

def main():
    args = parse_args()
    print(f"Rarefying {args.input_table} and outputting to {args.output}")

    # Define rarefaction depths
    rdepth_wgs = 500000  # Depth to rarefy shotgun metagenomic data
    rdepth_amp = 10000  # Depth to rarefy amplicon data

    # Determine the rarefaction depth based on sequencing type
    depth = rdepth_wgs if args.seq_type == 'wgs' else rdepth_amp

    # Read in the data
    M = pd.read_csv(args.input_table, sep=",", index_col=0)

    # Discard any sample (column) that does not meet the rarefaction depth
    M = M.loc[:, M.sum(axis=0) > depth]

    # Save column (sample) names for later
    sample_names = M.columns

    # Save the row (taxa) names for later
    taxa_names = M.index

    # Transpose data frame so samples are rows
    Mt = M.transpose()

    # Convert the dataframe to a numpy array for rarefaction
    Mt_array = Mt.to_numpy()

    # Rarefy the array
    Mt_array_rarefied = rarefaction(Mt_array, depth, seed=1)

    # Check that sample sums are all now the required rarefaction depth
    print(np.sum(Mt_array, axis=1))  # original depths
    print(np.sum(Mt_array_rarefied, axis=1))  # rarefied depths

    # Now add the sample and taxa names back to the dataframe
    Mt_array_rarefied_df = pd.DataFrame(Mt_array_rarefied, index=sample_names, columns=taxa_names)

    # Transpose again to get data into original format
    Mt_array_rarefied_df_t = Mt_array_rarefied_df.transpose()

    # Save the rarefied table
    file_path = Path(args.output)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    Mt_array_rarefied_df_t.to_csv(file_path, index=True)

if __name__ == "__main__":
    main()
