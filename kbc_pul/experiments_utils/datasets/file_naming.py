import os

from kbc_pul.experiments_utils.datasets.binary_literal_entity_enum import BinaryLiteralEntity


def get_edge_count_csv_filename_for(file_dir: str,
                                    predicate_name: str,
                                    entity_position: BinaryLiteralEntity) -> str:
    return os.path.join(file_dir, f"{predicate_name}_n_edges_{entity_position.value}.csv")


def get_edge_count_kb_filename_for(file_dir: str,
                                   predicate_name: str,
                                   entity_position: BinaryLiteralEntity) -> str:
    return os.path.join(file_dir, f"{predicate_name}_n_edges_{entity_position.value}.pl")


edge_counts_dir_base: str = 'observed_edge_counts'


def get_edge_count_dir(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, edge_counts_dir_base)
