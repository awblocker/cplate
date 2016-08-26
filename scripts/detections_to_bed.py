"""Converts cplate detections to a BED file."""
import argparse
import os
import re

import pandas as pd

COLUMNS = ['chrom', 'start', 'end', 'name', 'score', 'strand']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detections', dest='detections',
                        help='cplate detections file')
    parser.add_argument('--gene_number', dest='gene_number', default=-1,
                        type=int, help='row number for gene')
    parser.add_argument('--genes', dest='genes', help='gene definition file')
    parser.add_argument('--output', dest='output', default='',
                        help='output BED file')
    return parser.parse_args()


def main():
    args = parse_args()
    detections = pd.read_table(args.detections, delimiter=' ')

    # Infer gene number if needed.
    gene_number = args.gene_number
    if gene_number < 0:
        gene_number = (
            int(re.search(r'gene(\d+)', args.detections).group(1)) - 1)

    # Infer output path if needed.
    output = args.output
    if output == '':
        output = os.path.splitext(args.detections)[0] + '.bed'

    genes = pd.read_csv(args.genes)
    gene = genes.iloc[gene_number]
    intervals = []
    for _, row in detections.iterrows():
        start = int(row['pos'] - (row['n'] - 1) / 2.)
        end = int(row['pos'] + (row['n'] - 1) / 2.) + 1
        interval = {'start': start,
                    'end': end,
                    'name': '.',
                    'score': 0,
                    'strand': '.',
                   }
        intervals.append(interval)
    intervals = pd.DataFrame(intervals)
    intervals['chrom'] = gene['Chrom']
    intervals['start'] = intervals['start'] + gene['Start']
    intervals['end'] = intervals['end'] + gene['Start']
    intervals = intervals[COLUMNS]
    intervals.to_csv(output, sep='\t', header=False, index=False, quoting=False)


if __name__ == '__main__':
    main()
