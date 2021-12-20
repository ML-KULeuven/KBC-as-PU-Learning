import csv
from typing import List, Set

from kbc_pul.prolog_utils.prolog_token_encoder import TokenEncoder
from kbc_pul.experiments_utils.datasets.is_number_type_check import is_number


def tokenize_line_with_angle_brackets(line) -> List[str]:
    tokens = []

    current_start = 0
    is_opened = False
    for i in range(len(line)):
        if line[i] == "<":
            if is_opened:
                raise Exception("Found < when already reading token; nesting not supported.")
            else:
                is_opened = True
                current_start = i + 1
        if line[i] == ">":
            if not is_opened:
                raise Exception("Found > when not reading token; something went wrong.")
            else:
                is_opened = False
                token = line[current_start:i]
                tokens.append(token)
    return tokens


def cut_of_angle_brackets(token: str) -> str:
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    else:
        if "<" in token:
            raise Exception("Unclosed < in ", token)
        elif ">" in token:
            raise Exception("Unclosed > in ", token)
        else:
            return token


def clean_row(row,
              token_encoder: TokenEncoder,
              rels_with_numerical_e1: Set[str],
              rels_with_numerical_e2: Set[str],
              check_for_numerical_entities: bool = False
              ) -> List:
    if len(row) != 3:
        print(f"Row has length {len(row)} != 3:\n\t{row}")
        row = " ".join(row)
        row = tokenize_line_with_angle_brackets(row)
        if len(row) != 3:
            raise Exception(f"Row has length {len(row)} != 3: \n {row}")

    # no numbers as entities
    if check_for_numerical_entities:
        e1 = row[0]
        rel = row[1]
        e2 = row[2]
        if rel not in rels_with_numerical_e1:
            if is_number(e1):
                print(f"{rel} has some numerical e1's, e.g. {e1}")
                rels_with_numerical_e1.add(rel)

        if rel not in rels_with_numerical_e2:
            if is_number(e2):
                print(f"{rel} has some numerical e2's, e.g. {e2}")
                rels_with_numerical_e1.add(rel)

    cleaned_row = [
        token_encoder.encode(cut_of_angle_brackets(row[0]), is_entity=True),
        token_encoder.encode(cut_of_angle_brackets(row[1]), is_entity=False),
        token_encoder.encode(cut_of_angle_brackets(row[2]), is_entity=True)
    ]
    return cleaned_row


def convert_tsv_to_csv(
        original_tsv_with_angle_brackets_filename: str, output_csv_filename,
        check_for_numerical_entities: bool = False
):
    token_encoder = TokenEncoder(should_cache=False)

    rels_with_numerical_e1 = set()
    rels_with_numerical_e2 = set()

    with open(original_tsv_with_angle_brackets_filename, 'r') as tsv_ifile, open(output_csv_filename, "w") as csv_ofile:
        csv_reader = csv.reader(tsv_ifile, delimiter="\t")
        csv_writer = csv.writer(csv_ofile)
        for row in csv_reader:

            cleaned_row = clean_row(
                row=row,
                token_encoder=token_encoder,
                rels_with_numerical_e1=rels_with_numerical_e1,
                rels_with_numerical_e2=rels_with_numerical_e2,
                check_for_numerical_entities=check_for_numerical_entities
            )

            csv_writer.writerow(cleaned_row)

    print("DONE CLEANING")
