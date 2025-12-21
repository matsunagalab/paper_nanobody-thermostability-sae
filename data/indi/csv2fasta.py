#!/usr/bin/env python3
"""
csv2fasta.py  –  Convert NbThermo_train.csv to FASTA

Usage:
    python csv2fasta.py NbThermo_train.csv            # writes NbThermo_train.fasta
    python csv2fasta.py NbThermo_train.csv -o out.fa  # custom output file
"""
import argparse
import csv
import os
import textwrap

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NbThermo CSV to FASTA.")
    parser.add_argument("csv_file", help="Input CSV file (e.g. NbThermo_train.csv)")
    parser.add_argument("-o", "--output", help="Output FASTA file (default: <csv_basename>.fasta)")
    args = parser.parse_args()

    out_path = args.output or os.path.splitext(args.csv_file)[0] + ".fasta"

    with open(args.csv_file, newline="") as csvfile, open(out_path, "w") as fasta:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader, start=1):
            seq   = row["sequence"].strip()
            tm    = row["DB"].strip()
            leng  = row["Length"].strip()

            header = f">seq{idx}|DB={tm}|Len={leng}"
            fasta.write(header + "\n")
            fasta.write("\n".join(textwrap.wrap(seq, 60)) + "\n")

    print(f"Wrote {idx} sequences to '{out_path}'")

if __name__ == "__main__":
    main()


