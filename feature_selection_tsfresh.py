# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

from absl import logging
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters


def feature_extraction(df_x,col_n,standard_parameters,n_jobs,fc_parameters=None,kind_parameters=None,chunksize=None):
    """Runs tsfresh feature_extraction for pandas Dataframes

    Args:
        df_x (pd.DataFrame): Pandas Dataframe with feature values
        col_n (dict): Dictionary that contains column_id, column_sort, column_kind, and optional column_value
        standard_parameters (string): string of one of the tsfresh defaults parameter dictionaries
        n_jobs (int): Amount of processors that will be used for Parallelization
        fc_parameters (dict, optional): mapping from feature calculator names to parameters. Only those names which are keys in this dict will be calculated. Defaults to None.
        kind_parameters (dict, optional):  mapping from kind names to objects of the same type as the ones for default_fc_parameters. If you put a kind as a key here, the fc_parameters object (which is the value), will be used instead of the default_fc_parameters. Defaults to None.
        chunksize (int, optional): The size of one chunk that is submitted to the worker process for the parallelisation. Defaults to None.
    Returns:
        [pd.DataFrame]: return pandas dataframe with extracted features
    """  
    if fc_parameters:
        settings = eval(fc_parameters)
    elif standard_parameters == 'ComprehensiveFCParameters':
        settings = ComprehensiveFCParameters()
    elif standard_parameters == 'MinimalFCParameters':
        settings = MinimalFCParameters()
    elif standard_parameters == 'EfficientFCParameters':
        settings = EfficientFCParameters()

    if kind_parameters:
        kind_parameters = eval(kind_parameters)

    df_x_extracted = extract_features(timeseries_container=df_x,
                                        column_id=col_n['column_id'],
                                        column_sort=col_n['column_sort'],
                                        column_kind=col_n['column_kind'],
                                        column_value=col_n['column_value'],
                                        default_fc_parameters=settings,
                                        kind_to_fc_parameters=kind_parameters,
                                        n_jobs=n_jobs,
                                        chunksize=chunksize)
    logging.info('Columns before feature extraction \n {} \n '.format(df_x.columns.to_list()))
    logging.info('Columns after feature extraction \n {} \n '.format(df_x_extracted.columns.to_list()))
    return df_x_extracted

def feature_selection(df_x,df_y,fdr_level,n_jobs,chunksize=None,target_col=None):
    """Runs tsfresh feature_selection for pandas Dataframes
    Args:
        df_x (pd.DataFrame): Pandas Dataframe with feature values
        df_y (pd.DataFrame): Pandas Dataframe with only one column or when target_col is defined, with target_col.
        fdr_level (float): The FDR level that should be respected, this is the theoretical expected percentage of irrelevant features among all created features.
        n_jobs (int): Amount of processors that will be used for Parallelization
        chunksize (int, optional): The size of one chunk that is submitted to the worker process for the parallelisation. Defaults to None.
        target_col (string, optional): Column Name of target column which is needed to test which features are relevant. Defaults to None.
    Returns:
        [pd.DataFrame]: return pandas dataframe with selected features
    """    
    assert df_x.shape[0] == df_y.shape[0], 'Feature and targets amount of rows not equal!'
    if df_y.shape[1] > 1:
        df_y_series = pd.Series(data=df_y[target_col].values)
    else:
        df_y_series = pd.Series(data=df_y.values)

    df_x_selected = select_features(df_x,df_y_series,fdr_level=fdr_level,n_jobs=n_jobs,show_warnings=True,chunksize=chunksize)

    logging.info('Columns before feature selection \n {} \n '.format(df_x.columns.to_list()))
    logging.info('Columns after feature selection \n {} \n '.format(df_x_selected.columns.to_list()))
    return df_x_selected


def remove_columns(df_x,kind_parameters):
    """Runs 'feature selection' with given parameter configuration. More precisely this fuction removes columns that are not in the configuration dict

    Args:
        df_x (pd.DataFrame): Input Dataframe
        kind_parameters (dict): dict which includes relevant columnnames to keep when a given df_x should be transformed

    Returns:
        [pd.DataFrame]: Dataframe without irrelevant /  with relevant columns
    """
    columns_to_drop = []
    for col in df_x.columns.to_list():
        if col not in kind_parameters:
            columns_to_drop.append(col)

    logging.info('Columns before feature selection \n {} \n '.format(df_x.columns.to_list()))
    df_x.drop(columns=columns_to_drop,inplace=True)
    logging.info('Columns after feature selection \n {} \n '.format(df_x.columns.to_list()))

    return df_x
