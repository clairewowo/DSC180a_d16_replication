#!/usr/bin/env python3

# conda install -c conda-forge -c bioconda vsearch

# OUTPUTS ARE NOW IN V6_TABLES
"""V6 FASTA -> 24-Order table (Greengenes 13_8, SINTAX). Needs: vsearch, pandas."""
import os, subprocess, urllib.request, tarfile, collections
import pandas as pd

FASTA   = "/home/clw009/private/DSC180A/data/01_sequence_data/fasta-1794098140692_ICM_PML_Bv6.fa"
THREADS = 8
CUTOFF  = 0.5          # SINTAX confidence; V6 is short, so 0.8 will leave many reads unassigned
GG_URL  = "https://gg-sg-web.s3-us-west-2.amazonaws.com/downloads/greengenes_database/gg_13_5/gg_13_8_otus.tar.gz"


# orders that we are interested in, going into MAP
ORDERS_24 = [
    "Actinomycetales", "Alteromonadales", "Bacillales", "Burkholderiales",
    "Cyanobacteria", "Desulfobacterales", "Enterobacteriales", "Flavobacteriales",
    "Legionellales", "Methylophilales", "Oceanospirillales", "Opitutales",
    "Planctomycetales", "Prochlorales", "Pseudomonadales", "Rhizobiales",
    "Rhodobacterales", "Rhodospirillales", "Rickettsiales", "Sphingobacteriales",
    "Sphingomonadales", "Verrucomicrobiales", "Vibrionales", "Xanthomonadales",
]
RENORM_TO_24 = True   # False: fractions of all cleaned bacterial reads; True: fractions within the 24 Orders


def parse_header(h):
    """>5267989|ICM_PML_Bv6--PML_0001_2007_02_16|13  ->  ('PML_0001_2007_02_16', 13)"""
    rid, ds, n = h[1:].split("|")
    return ds.split("--")[1], int(n)


# ---------- 1. Greengenes reference -> SINTAX format ----------
if not os.path.exists("gg_13_8_otus"):
    urllib.request.urlretrieve(GG_URL, "gg.tar.gz")
    tarfile.open("gg.tar.gz").extractall()

tax = {}
with open("gg_13_8_otus/taxonomy/99_otu_taxonomy.txt") as f:
    for line in f:
        i, t = line.rstrip("\n").split("\t")[:2]
        tax[i] = t

with open("gg_sintax.fa", "w") as out, open("gg_13_8_otus/rep_set/99_otus.fasta") as f:
    keep = False
    for line in f:
        if line.startswith(">"):
            i = line[1:].split()[0]
            parts = [p.strip() for p in tax.get(i, "").split(";") if len(p.strip()) > 3]
            keep = bool(parts)
            if keep:
                lab = ",".join(f"{p[0]}:{p[3:].replace(',', '')}" for p in parts)
                out.write(f">{i};tax={lab};\n")
        elif keep:
            out.write(line)

# ---------- 2. Parse reads: unique sequences + per-sample counts ----------
counts = collections.defaultdict(collections.Counter)   # seq -> {sample: n}
h = seq = None


def flush():
    if h is not None:
        samp, n = parse_header(h)
        counts[seq.upper().replace("U", "T")][samp] += n


with open(FASTA) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            flush()
            h, seq = line, ""
        else:
            seq += line
    flush()

print(f"{len(counts)} unique sequences")
with open("unique.fa", "w") as out:
    for n, s in enumerate(counts):
        out.write(f">u{n}\n{s}\n")
useq = {f"u{n}": s for n, s in enumerate(counts)}

# ---------- 3. Classify ----------
subprocess.run(["vsearch", "--sintax", "unique.fa", "--db", "gg_sintax.fa",
                "--tabbedout", "sintax.tsv", "--sintax_cutoff", str(CUTOFF),
                "--strand", "both", "--threads", str(THREADS)], check=True)

# ---------- 4. Build table, clean up ----------
rows = []
with open("sintax.tsv") as f:
    for line in f:
        c = line.rstrip("\n").split("\t")
        q, passed = c[0], (c[3] if len(c) > 3 else "")
        r = dict(x.split(":", 1) for x in passed.split(",") if ":" in x)
        phylum, cls, order = r.get("p"), r.get("c"), r.get("o")
        if cls == "Chloroplast":
            lab = "CHLOROPLAST"
        elif phylum == "Cyanobacteria":
            lab = "Cyanobacteria"            # p__Cyanobacteria minus Chloroplast (Prochlorales folded in)
        elif r.get("k") != "Bacteria":
            lab = "NON_BACTERIA"             # archaea/unclassified
        elif order:
            lab = order
        else:
            lab = "unassigned"
        for samp, n in counts[useq[q]].items():
            rows.append((lab, samp, n))

df = pd.DataFrame(rows, columns=["order", "sample", "n"])
tab = df.pivot_table(index="order", columns="sample", values="n", aggfunc="sum", fill_value=0)
tab.to_csv("V6_order_by_sample_raw.tsv", sep="\t")

chl = tab.loc["CHLOROPLAST"].sum() / tab.values.sum() if "CHLOROPLAST" in tab.index else 0
print(f"chloroplast fraction: {chl:.1%}")

clean = tab.drop(index=[i for i in ["CHLOROPLAST", "NON_BACTERIA", "unassigned"] if i in tab.index])
rel = clean / clean.sum()                     # renormalise after removals
rel.to_csv("V6_order_by_sample_relabund.tsv", sep="\t")

# ---------- 5. Keep only the paper's 24 Orders ----------
missing = [o for o in ORDERS_24 if o not in clean.index]
print("Of the 24, not found in this data (set to 0):", missing)   # expect Prochlorales (folded into Cyanobacteria)

sub = clean.reindex(ORDERS_24, fill_value=0)
rel24 = sub / (sub.sum() if RENORM_TO_24 else clean.sum())
rel24.to_csv("V6_24orders_by_sample_relabund.tsv", sep="\t")
print("Reads per sample in the 24 Orders:\n", sub.sum().sort_values().head(10))
print("Fraction of cleaned reads covered by the 24 Orders:\n", (sub.sum() / clean.sum()).describe())

# ---------- 6. Dates, then average samples sharing a date ----------
meta = pd.DataFrame({"sample": rel24.columns})
meta["date"] = pd.to_datetime(meta["sample"].str.extract(r"(\d{4}_\d{2}_\d{2})$")[0], format="%Y_%m_%d")
print(meta.sort_values("date").to_string())
print("samples:", len(meta), "| unique dates:", meta["date"].nunique())

by_date = rel24.T.join(meta.set_index("sample")["date"]).groupby("date").mean().T
by_date.to_csv("V6_24orders_by_date_relabund.tsv", sep="\t")
print("Wrote V6_24orders_by_date_relabund.tsv")
