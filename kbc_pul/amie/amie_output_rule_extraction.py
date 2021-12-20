"""
Parsing raw output and writing it as a TSV file.
Reading the TSV file out as a list of RuleWrappers
"""

from typing import List
import pandas as pd

from kbc_pul.data_structures.rule_wrapper import RuleWrapper


def create_rule_tsv_file_from_amie_output_file(amie_output_filename: str, amie_rules_tsv_filename: str) -> None:
    """
    Extract from the AMIE output the lines describing rules.

    Reason: AMIE's output also contains info about the mining process itself, which we do not need.
    :param amie_output_filename:
    :param amie_rules_tsv_filename:
    :return:
    """
    with open(amie_output_filename, 'r') as amie_output_file, open(amie_rules_tsv_filename, 'w') as amie_rules_file:
        should_write_lines: bool = False
        for line in amie_output_file:
            if not should_write_lines:
                if line.startswith("Starting the mining phase"):
                    should_write_lines = True
            else:
                if line.startswith("Mining done"):
                    should_write_lines = False
                else:
                    amie_rules_file.write(line)


def get_amie_rules_from_rule_tsv_file(amie_rules_tsv_filename: str) -> List[RuleWrapper]:
    """
    Return a list of rules represented as RuleWrappers from a TSV file separated using a TAB.

    :param amie_rules_tsv_filename: TSV file containing parsed AMIE output
    :return: List of rules represented as RuleWrapper objects
    """
    rules_df = pd.read_csv(amie_rules_tsv_filename, sep="\t")

    rule_wrapper_list: List[RuleWrapper] = [
        RuleWrapper.create_rule_wrapper_from(row_series)
        for row_index, row_series in rules_df.iterrows()
    ]
    return rule_wrapper_list
