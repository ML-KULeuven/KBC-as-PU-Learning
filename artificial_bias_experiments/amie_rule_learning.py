"""
AMIE rules for use in experiments_utils

"""
import os

from kbc_pul.amie.amie_output_rule_extraction import create_rule_tsv_file_from_amie_output_file
from kbc_pul.amie.amie_wrapper import run_amie_parametrized
from kbc_pul.project_info import data_dir as kbc_e_metric_data_dir


def get_amie_rules_output_dir(
        dataset_name: str
) -> str:
    """
    Return directory name to store anything related to AMIE's rules.

    E.g. '$PROJECT_DATA_DIR/data/artificial_bias_experiments/yago3_10/amie'

    :param dataset_name: e.g. "yago3_10"
    :return: file name path to store anything related to AMIE's rules.
    """
    amie_rules_output_dir: str = os.path.join(
        kbc_e_metric_data_dir,
        'artificial_bias_experiments',
        dataset_name,
        "amie"
    )
    return amie_rules_output_dir


def get_amie_rule_tsv_filename(
        filename_ground_truth_dataset: str,
        dataset_name: str,
        min_std_confidence: float
) -> str:
    """
    Returns the TSV filename containing AMIE rules for
        the given dataset and
        the given minimum standard confidence (i.e. CWA confidence).

    IF the rule TSV file does not yet exist,
    THEN:
        * the amie dir is created if it does not yet exist,
            e.g. '$PROJECT_DATA_DIR/data/artificial_bias_experiments/yago3_10/amie'
        * amie is run, and its output written to file,
            e.g. '$PROJECT_DATA_DIR/data/artificial_bias_experiments/yago3_10/amie/yago3_10_output.txt'

        * the amie tsv filename is created


    :param filename_ground_truth_dataset: CSV file containing the triples to mine rules on
    :param dataset_name: str: name of the dataset, e.g. 'yago3_10'
    :param min_std_confidence: float: mimimum value for the CWA/standard confidence of mined rules e.g. 0.1
    :return:
    """
    amie_rules_output_dir = get_amie_rules_output_dir(dataset_name=dataset_name)

    if not os.path.exists(amie_rules_output_dir):
        os.makedirs(amie_rules_output_dir)
        print(f"Made amie rule dir at: {amie_rules_output_dir}")

    amie_rule_tsv_filename = os.path.join(
        amie_rules_output_dir,
        f"{dataset_name}_amie_rules_min_std_conf{min_std_confidence}.tsv"
    )

    if not os.path.exists(amie_rule_tsv_filename):
        print("Learn rules with amie")
        amie_output_filename = os.path.join(
            amie_rules_output_dir,
            f"{dataset_name}_amie_output.txt"
        )
        run_amie_parametrized(
            amie_tsv_input_filename=filename_ground_truth_dataset,
            amie_output_filename=amie_output_filename,
            datalog_notation_for_rules=True,
            min_std_confidence=min_std_confidence
        )
        create_rule_tsv_file_from_amie_output_file(
            amie_output_filename=amie_output_filename,
            amie_rules_tsv_filename=amie_rule_tsv_filename
        )
        print(f"Find rule TSV in {amie_rule_tsv_filename}")
        return amie_rule_tsv_filename
    else:
        return amie_rule_tsv_filename
