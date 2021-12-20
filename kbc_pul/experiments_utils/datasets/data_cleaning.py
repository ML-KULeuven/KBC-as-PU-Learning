import csv
from typing import List

import numpy as np
from tqdm import tqdm
from kbc_pul.experiments_utils.datasets.data_cleaning_helper import clean_row

from kbc_pul.prolog_utils.prolog_token_encoder import TokenEncoder


def clean_triples(dataset_part: np.ndarray, dataset_part_output_csv_filename: str, should_sort: bool, separator:str):
    token_encoder = TokenEncoder(should_cache=False)
    rels_with_numerical_e1 = set()
    rels_with_numerical_e2 = set()

    # cleaned_to_original_map = dict()
    #
    # row_set = set()
    with open(dataset_part_output_csv_filename, 'w') as csv_ofile:
        csv_writer = csv.writer(csv_ofile, delimiter=separator)

        if should_sort:
            print("Sorting on Object...")
            dataset_part = dataset_part[dataset_part[:, 2].argsort()]  # First sort doesn't need to be stable.
            print("Sorting on Subject...")
            dataset_part = dataset_part[dataset_part[:, 0].argsort(kind='mergesort')]
            print("Sorting on Rel...")
            dataset_part = dataset_part[dataset_part[:, 1].argsort(kind='mergesort')]


        dataset_length = len(dataset_part)
        row: List[str]
        for row_index, row in tqdm(enumerate(dataset_part), total=dataset_length):
            # row_tup = tuple(row)
            # if row_tup in row_set:
            #     print("Duplicate row:", row_tup)
            # else:
            #     row_set.add(row_tup)
            cleaned_row = clean_row(row, token_encoder=token_encoder,
                                    rels_with_numerical_e1=rels_with_numerical_e1,
                                    rels_with_numerical_e2=rels_with_numerical_e2,
                                    check_for_numerical_entities=True
                                    )
            # cleaned_row_tup = tuple(cleaned_row)
            # if cleaned_row_tup in cleaned_to_original_map:
            #     raise Exception("Duplicate row:\n"
            #                     f"{cleaned_row} maps to both\n"
            #                     f"\t{cleaned_to_original_map[cleaned_row_tup]}   and\n"
            #                     f"\t{row_tup}")
            # cleaned_to_original_map[cleaned_row_tup] = row_tup
            # print(f"Original row: {row}")
            # print(f"Cleaned row: {cleaned_row}\n")

            # if row_index >=10:
            #     break
            csv_writer.writerow(cleaned_row)
