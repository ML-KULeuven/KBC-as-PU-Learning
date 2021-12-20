"""
Python wrapper for AMIE3.

"""
import json
import os

import subprocess
from typing import Optional, List, Dict
from kbc_pul.project_info import amie_jar_settings_json


def _get_amie_jar_filename_from_amie_dir_dict() -> str:
    """
    Returns AMIE's jar filename from the a JSON settings file:
    {
        "amie_dir": "external/AMIE3",
        "amie_jar_filename": "amie-dev.jar"
    }

    :return: AMIE's jar filename from the a JSON settings file
    """
    if not os.path.exists(amie_jar_settings_json):
        raise Exception(f"No AMIE specification found at {amie_jar_settings_json}")

    with open(amie_jar_settings_json, "r") as ace_root_file:
        ace_root_dict: Dict[str, str] = json.load(ace_root_file)

    amie_root_dir: str = ace_root_dict["amie_dir"]
    amie_jar_filename: str = ace_root_dict["amie_jar_filename"]
    return os.path.join(amie_root_dir, amie_jar_filename)


def _run_amie(amie_tsv_input_filename: str,
              amie_output_filename: str,
              amie_options_list: Optional[List[str]] = None,
              amie_jar_filename: Optional[str] = None,
              work_dir: Optional[str] = None,
              verbose: bool = False
              ):
    """
    Should not be called directly. See also: run_amie_parametrized()

    :param amie_tsv_input_filename:
    :param amie_output_filename:
    :param amie_options_list:
    :param amie_jar_filename:
    :param work_dir:
    :param verbose:
    :return:
    """
    if amie_jar_filename is None:

        default_amie_jar_absolute_filename = _get_amie_jar_filename_from_amie_dir_dict()
        if not os.path.exists(default_amie_jar_absolute_filename):
            raise Exception(f"No AMIE jar-file given & cannot find default jar at {default_amie_jar_absolute_filename}")
        else:
            amie_jar_filename = default_amie_jar_absolute_filename
    else:
        if not os.path.exists(amie_jar_filename):
            raise Exception(f"Cannot find AMIE jar-file at {amie_jar_filename}")

    if amie_options_list is not None:
        options_str = " ".join(amie_options_list)
    else:
        options_str = ""

    output_file_command = " 2>&1 | tee " + amie_output_filename

    command = "java -jar " + amie_jar_filename + " " + options_str + ' ' + amie_tsv_input_filename + output_file_command

    if work_dir is None:
        work_dir = os.getcwd()
    if verbose:
        print(f"Running AMIE from work dir: {work_dir}")

    print(command)
    process_code = subprocess.call(command,
                                   shell=True,
                                   cwd=work_dir)
    return process_code


def run_amie_parametrized(
        amie_tsv_input_filename: str,
        amie_output_filename: str,
        datalog_notation_for_rules: bool,
        max_n_atoms_in_rule: Optional[int] = None,
        min_std_confidence: Optional[float] = None,
        min_pca_confidence: Optional[float] = None,
        min_head_coverage: Optional[float] = None,
        min_head_relation_initial_size: Optional[None] = None,
        min_absolute_support: Optional[int] = None,
        delimiter: Optional[str] = None,
        enable_constants: bool = False,
        amie_options_list: Optional[List[str]] = None,
        amie_jar_filename: Optional[str] = None,
        work_dir: Optional[str] = None,
        verbose: bool = False

) -> int:
    """

    :param amie_tsv_input_filename: TSV-like input file, with one row per fact
    :param amie_output_filename: File to write AMIE's output to
    :param max_n_atoms_in_rule: Max. nb of atoms in the antecedent and consequent of rules.  Default: 3

    :param min_std_confidence: Minimum standard confidence threshold. Default: 0.0.
        This value is not used for pruning, only for filtering of the results.
    :param min_pca_confidence: Minimum PCA confidence threshold. Default: 0.0.
        This value is not used for pruning, only for filtering of the results.

    :param min_absolute_support: Minimum absolute support. Default: 100 positive examples

    :param min_head_relation_initial_size: Minimum size of the relations to be considered as head relations.
        Default: 100 (facts or entities depending on the bias)
    :param min_head_coverage: Minimum head coverage. Default: 0.01

    :param enable_constants: Enable rules with constants. Default: false

    :param datalog_notation_for_rules: Print rules using the datalog notation. Default: false
    :param delimiter: Separator in input files (Default is TAB). E.g ","

    :param verbose:
    :param work_dir:
    :param amie_jar_filename:
    :param amie_options_list:

    :return: process_code: int  
    """
    extra_amie_options_list = []

    if datalog_notation_for_rules:
        extra_amie_options_list.append("-datalog")

    if max_n_atoms_in_rule is not None:
        extra_amie_options_list.append(f"-maxad {max_n_atoms_in_rule}")

    if min_std_confidence is not None:
        extra_amie_options_list.append(f"-minc {min_std_confidence}")
    if min_pca_confidence is not None:
        extra_amie_options_list.append(f"-minpca {min_pca_confidence}")

    if min_absolute_support is not None:
        extra_amie_options_list.append(f"-mins {min_absolute_support}")

    if min_head_relation_initial_size is not None:
        extra_amie_options_list.append(f"-minis {min_head_relation_initial_size}")

    if min_head_coverage is not None:
        extra_amie_options_list.append(f"-minhc {min_head_coverage}")

    if delimiter is not None:
        if isinstance(delimiter, str):
            extra_amie_options_list.append(f"-d {delimiter}")
        else:
            raise Exception(f"Unexpected type {type(delimiter)} for delimiter; expected str")

    if enable_constants:
        extra_amie_options_list.append("-const")

    if amie_options_list is not None:
        full_amie_options_list: List[str] = amie_options_list + extra_amie_options_list
    else:
        full_amie_options_list = extra_amie_options_list

    process_code: int = _run_amie(
        amie_tsv_input_filename=amie_tsv_input_filename,
        amie_output_filename=amie_output_filename,
        amie_options_list=full_amie_options_list,
        amie_jar_filename=amie_jar_filename,
        work_dir=work_dir,
        verbose=verbose
    )

    return process_code
