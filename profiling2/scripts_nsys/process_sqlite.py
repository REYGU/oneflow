#!/usr/bin/python

import sqlite3
import os
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", type=str)
# parser.add_argument("-t", "--target", type=str)
# parser.add_argument("--oplist", type=str)
args = parser.parse_args()

SQL_PATH = args.source
CSV_PATH = args.source.replace(".sqlite", ".raw.csv")

if not os.path.exists(SQL_PATH) and not os.path.exists(CSV_PATH):
    print(f"{args.source} not found...")
    exit()

if os.path.exists(SQL_PATH):
    conn = sqlite3.connect(SQL_PATH)
    c = conn.cursor()
    shape = os.path.basename(SQL_PATH).rsplit(".", 1)[0]

    cmd = f"select * from NVTX_EVENTS where text like 'model.net-%' or text like 'model-%';"
    cursor = conn.execute(cmd)
    headers = [i[0] for i in cursor.description]
    df = pd.DataFrame(columns=headers, data=list(cursor))
    df.insert(0, "time_ns", df["end"] - df["start"])
    df.insert(0, "shape", shape)
    df.insert(0, "op", df["text"].str.extract("model(?:.net)?-(\w+).+"))
    df.to_csv(CSV_PATH, index=False)
elif os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)

summary_headers = ["op", "shape", "time_ns", "text"]
summary = df[summary_headers].groupby(["op", "shape", "text"], as_index=False).mean()
summary["count"] = (
    df[summary_headers].groupby(["op", "shape", "text"], as_index=False).size()["size"]
)
summary.to_csv(SQL_PATH.replace(".sqlite", ".summary.csv"), index=False)

print("process success", SQL_PATH)
