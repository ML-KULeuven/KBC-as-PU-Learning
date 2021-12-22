# Unifying Knowledge Base Completion with PU Learning to Mitigate the Observation Bias

Source code related to the [AAAI22](https://aaai.org/Conferences/AAAI-22/) paper:

> Unifying Knowledge Base Completion with PU Learning to Mitigate the Observation Bias. 
> Jonas Schouterden, Jessa Bekker, Jesse Davis, Hendrik Blockeel. 

## Table of Contents

* [Abstract](https://github.com/ML-KULeuven/KBC-as-PU-Learning#abstract)
* [Installation](https://github.com/ML-KULeuven/KBC-as-PU-Learning#installation)
* [Notebooks](https://github.com/ML-KULeuven/KBC-as-PU-Learning#notebooks)
* [Running the experiemtns](https://github.com/ML-KULeuven/KBC-as-PU-Learning#running-the-experiments)
* [Generating the tables in the paper](https://github.com/ML-KULeuven/KBC-as-PU-Learning#generating-the-tables-in-the-paper)
* [Generating the images in the paper](https://github.com/ML-KULeuven/KBC-as-PU-Learning#generating-the-images-in-the-paper)
* [Preparation of the "ideal" Yago3-10 KB](https://github.com/ML-KULeuven/KBC-as-PU-Learning#preparation-of-the-ideal-yago3_10-kb)

## Abstract
The following is the abstract of our paper:

> Methods for Knowledge Base Completion (KBC) reason about a knowledge base (KB) in order to derive new facts that should be included in the KB. This is challenging for two reasons. First, KBs only contain positive examples. This complicates model evaluation which needs both positive and negative examples. Second, those facts that were selected to be included in the knowledge base, are most likely not an i.i.d. sample of the true facts, due to the way knowledge bases are constructed. In this paper, we focus on rule-based approaches, which traditionally address the first challenge by making assumptions that enable identifying negative examples, which in turn makes it possible to compute a rule’s confidence or precision. However, they largely ignore the second challenge, which means that their estimates of a rule’s confidence can be biased. This paper approaches rule-based KBC through the lens of PU learning, which can cope with both challenges. We make three contributions. (1) We provide a unifying view that formalizes the relationship between multiple existing confidences measures based on (i) what assumption they make about and (ii) how their accuracy depends on the selection mechanism. (2) We introduce two new confidence measures that can mitigate known biases by using propensity scores that quantify how likely a fact is to be included the KB. (3) We show through theoretical and empirical analysis that taking the bias into account improves the confidence estimates, even when the propensity scores are not known exactly.

## Installation

### Requirements

Create a fresh Python3 environment (3. or higher) and install the following packages:

* jupyter: for the notebooks.
* pandas: for representing the KB.
* problog : used for its parsing functionalty, i.e. parsing Prolog clauses from their string representation
* pylo2: see below
* matplotlib: plotting
* seaborn: plotting.
* tqdm: pretty status bars.
* unidecode: used when cleaning data.
* tabulate: for pretty table printouts
* dask.delayed and dask.distributed: for running experiments using dask 

### Installing Pylo2:

We use data structures from [Pylo2](https://github.com/sebdumancic/pylo2) to represent rules as Prolog clauses.
More specifically, Pylo2 data structures from `src/pylo/language/lp` are often used. 
To install Pylo2 in your Python environment, first clone it:
```shell
 git clone git@github.com:sebdumancic/pylo2.git
 cd pylo2
```
Note that Pylo has a lot of functionality we don't need. 
As we don't Pylo´s bindings to Prolog engines, we don't need those bindings. 
To install Pylo2 without these bindings, modify its `setup.py` by ading right before the line:
```python
print(f"Building:\n\tGNU:{build_gnu}\n\tXSB:{build_xsb}\n\tSWIPL:{build_swi}")
``` 
the following lines:
```python
build_gnu = None
build_xsb = None
build_swi = None
```
Then, install Pylo in the current environment using
```shell
python setup.py install
```

## Notebooks

Different notebooks are provided:
* [How to run AMIE from Python](./notebooks/amie_general)
* [The yago3-10 dataset: cleaning & exploration](./notebooks/amie_general)
* [How to apply a rule to a KB](./notebooks/pandas_rule_evaluation/how_to_apply_a_rule_to_a_pandas_kb.ipynb)

## Running the experiments

For a description on how to run the experiments, see [here](./notes/how-to-run-the-experiments.md).

## Generating the tables in the paper

For instructions to generate the tables in the paper from the results, see [here](./notes/how-to-generate-tables-in-paper.md).


## Generating the images in the paper

Instructions on how to generate the images in the paper can be found [here](./notes/how-to-generate-images-in-paper.md).

## Preparation of the "ideal" Yago3_10 KB

In the paper, the experiments are run on a cleaned version of the yago3-10 datasets. 
The cleaning was done to remove unicode characters that might be incompatible with older prolog engines, 
using [./notebooks/yago3_10/data_exploration_and_preparation/yago3_10_data_cleaning.ipynb](./notebooks/yago3_10/data_exploration_and_preparation/yago3_10_data_cleaning.ipynb)

The original data was obtained using [AmpliGraph](https://docs.ampligraph.org/en/1.4.0/generated/ampligraph.datasets.load_yago3_10.html),
but can also be found under [./data/yago3_10/original](./data/yago3_10/original). 

The cleaned version can be found under [./data/yago3_10/cleaned_csv](./data/yago3_10/cleaned_csv). 




