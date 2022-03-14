# Tsfresh feature extraction and feature selection
Tsfresh is used to to extract characteristics from time series. \
The extracted features can be used to describe or cluster time series based on the extracted characteristics. Further, they can be used to build models that perform classification/regression tasks on the time series. Often the features give new insights into time series and their dynamics.


## Installation

Clone the repository and install all requirements using `pip install -r requirements.txt` .


## Usage

You can run the code in two ways.
1. Use command line flags as arguments `python main.py --input_path= --output_path=...`
2. Use a flagfile.txt which includes the arguments `python main.py --flagfile=example/flagfile.txt`

### Input Flags/Arguments

#### --input_path
Specify the a local or s3 object storage path where the dataframe is stored.
For a s3 object storage path a valid s3 configuration yaml file is required.

#### --output_path
Specify the path where the profile report will be stored.
For a s3 object storage path a valid s3 configuration yaml file is required.

#### --stage
Specify method that will be used.

#### --stage_config
Use fit when you run the stage for the first time one a dataset. Use transform when you want to run the stage with a already present/fitted configuration.

#### --config_filename
Filetype json required. Name of the configuration file that will be saved as json when stage_config is fit. When stage_config is transform this should be the name of the configuration file to load.

#### --filenames_x
List of feature(_x) filenames for feature selection/extraction.

#### --filenames_y
List of target(_y) filenames for feature selection.

#### --fdr_level
The FDR level that should be respected, this is the theoretical expected percentage of irrelevant features among all created features.

#### --n_jobs
Number of processes to use during the p-value calculation, set 0 to disable Parallelization

#### --chunksize
The size of one chunk that is submitted to the worker process for the parallelisation. The chunksize can have an crucial influence on the optimal cluster performance and should be optimised in benchmarks for the problem at hand.

#### --column_names

| ID | time | kind | value |
|---|---|---|---|
| Bauteil_1 | 10  | Strom_spindel |  1.1 |
| Bauteil_1  | 20  | Strom_spindel  | 1.2  |
| Bauteil_1  | 30  | Strom_spindel |   | 1.1  |
| Bauteil_1  | 10  | Spannung_spindel  |  14 |
| Bauteil_1  | 20  | Spannung_spindel  |  14 |
| Bauteil_1  | 30  | Spannung_spindel  |  10 |
| Bauteil_2   | 10  | Strom_spindel  | 2.4  |
| Bauteil_2  | 20  | Strom_spindel  | 1.1  |
| Bauteil_2  | 30  | Strom_spindel  | 1.1 |
| Bauteil_2 | 10  | Spannung_spindel  | 12 |
| Bauteil_2 | 20  | Spannung_spindel  | 12  |
| Bauteil_2 | 30  | Spannung_spindel  | 11  |

"{'column_id':'id','column_sort':'time', 'column_kind':'', 'column_value':'values'}" \

[Input Option 2](https://tsfresh.readthedocs.io/en/latest/text/data_formats.html)

#### --standard_parameters
Select one of three tsfresh default parameter dictionary.
 - ComprehensiveFCParameters
 - MinimalFCParameters
 - EfficientFCParameters

https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html

#### --fc_parameters
Dict that defines individual function parameters for feature extraction.

#### --kind_parameters
Dict that defines individual kind parameters for feature extraction.

#### --y_target_column
If y shape[1] > 1 --> Column Name of target column which is needed to test which features are relevant. Target Vector can be binary or real-valued.


## Example

First move to the repository directory. 

We run e.g. select features for a new train dataset with `python main.py --flagfile=select_fit.txt`
Afterwards we run feature selection for a test dataset with the fitted configuration from the train step `python main.py --flagfile=select_transform.txt`

## Data Set

The data set was recorded with the help of the Festo Polymer GmbH. The features (`x.csv`) are either parameters explicitly set on the injection molding machine or recorded sensor values. The target value (`y.csv`) is a crucial length measured on the parts. We measured with a high precision coordinate-measuring machine at the Laboratory for Machine Tools (WZL) at RWTH Aachen University.