import csv
import os
import sqlite3
import sys

import numpy as np


def create_database(shard_name: str):
    """
    If the databse does not exist, create it
    TODO: update so that we can take in multiple shards
    """
    db_filepath = f"data/extracted_databases/{shard_name}.db"
    if os.path.exists(db_filepath):
        print("Database already exists")
        return

    print("Creating database")
    connection = sqlite3.connect(db_filepath)
    cur = connection.cursor()

    cur.execute(
        """CREATE TABLE ngrams(
            shard INT,
            document INT,
            ngrams INT
        );"""
    )
    ngrams_csv_filepath = f"data/extracted_databases/{shard_name}.ngrams.csv"
    with open(ngrams_csv_filepath) as ngrams_csv:
        rows = csv.reader(ngrams_csv)
        cur.executemany("INSERT INTO ngrams VALUES (?, ?, ?)", rows)
    print("Added ngrams to the databse.")
    cur.execute(
        """
        CREATE INDEX ngrams_index
        ON ngrams (ngrams);
        """
    )
    print("Created ngrams index.")

    cur.execute(
        """CREATE TABLE urls(
            document INT,
            url TEXT
        );"""
    )  # Could add some foreign key constraints to link the two tables
    urls_csv_filepath = f"data/extracted_databases/{shard_name}.urls.csv"
    with open(urls_csv_filepath) as urls_csv:
        rows = csv.reader(urls_csv)
        cur.executemany("INSERT INTO urls VALUES (?, ?)", rows)
    print("Added urls to the databse.")

    csv.field_size_limit(sys.maxsize)
    # To avoid running into `_csv.Error: field larger than field limit (131072)` error
    # https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
    cur.execute(
        """CREATE TABLE htmls(
            document INT,
            html TEXT
        );"""
    )
    htmls_csv_filepath = f"data/extracted_databases/{shard_name}.htmls.csv"
    with open(htmls_csv_filepath) as htmls_csv:
        rows = csv.reader(htmls_csv)
        cur.executemany("INSERT INTO htmls VALUES (?, ?)", rows)
    print("Added htmls to the databse.")

    connection.commit()
    connection.close()
    print("Database created.")


# n-grams computation

# Mapping each uni, bi, tri, 4-grams into a unique id.
# Let's call V the size of the vocabulary.
# For instance, sequence of integers is [a, b, c] -> a V**2 + b V + c

# 1-grams \in [|1; V|]
# 2-grams \in [|V + 1; V**2 + V|]
# 3-grams \in [|V**2 + V + 1; V**3 + V**2 + V|]
# 4-grams \in [|V**3 + V**2 + V + 1; V**4 + V**3 + V**2 + V|]

# for V=50_257,
# V**4 + V**3 + V**2 + V = 6_379_621_074_231_211_300
# int64 upper bound =      9_223_372_036_854_775_807


def unique_tolist(l_):
    return np.unique(l_).tolist()


def get_ngrams(ids: np.ndarray, voc_size: int, n: int):
    ids = ids + 1  # offset by one otherwise the mapping is not bijective...
    V_pow = [voc_size**i for i in range(4)]

    if len(ids) < n:
        raise ValueError

    # Horrible implementation but i don't want to mess it up yet, so going slowly
    # TODO: update
    if n == 1:
        return unique_tolist(ids)
    elif n == 2:
        bigrams = (V_pow[1] * ids)[:-1] + ids[1:]
        return unique_tolist(bigrams)
    elif n == 3:
        trigrams = (V_pow[2] * ids)[:-2] + (V_pow[1] * ids)[1:-1] + ids[2:]
        return unique_tolist(trigrams)
    elif n == 4:
        fourgrams = (V_pow[3] * ids)[:-3] + (V_pow[2] * ids)[1:-2] + (V_pow[1] * ids)[2:-1] + ids[3:]
        return unique_tolist(fourgrams)
    else:
        raise ValueError("n-grams for n>=5 are not supported yet.")
