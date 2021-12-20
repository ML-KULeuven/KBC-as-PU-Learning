import pandas as pd


def get_df_ground_truth(filename_ground_truth_dataset, sep: str) -> pd.DataFrame:
    """
    Reads a CSV file representing triples as a DataFrame with 3 columns: ["Subject", "Rel", "Object"] .

    :param filename_ground_truth_dataset:
    :param sep:
    :return:
    """
    df_ground_truth: pd.DataFrame = pd.read_csv(
        filename_ground_truth_dataset,
        names=["Subject", "Rel", "Object"],
        sep=sep
    )
    return df_ground_truth
