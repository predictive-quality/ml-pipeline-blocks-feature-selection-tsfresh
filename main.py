# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

from absl import logging, flags, app
from feature_selection_tsfresh import feature_extraction, feature_selection, remove_columns
from tsfresh import feature_extraction as feet
from s3_smart_open import get_filenames, read_pd_fth, to_json, to_pd_fth, read_json
import os

flags.DEFINE_string('input_path', default=None, help='path for input data')
flags.DEFINE_string('output_path', default=None, help='path for saving selected features')
flags.DEFINE_enum('stage',None,['extract','select','both'],'Specify method that will be used.')
flags.DEFINE_enum('stage_config','fit',['fit','transform'],'Use fit when you run the stage for the first time one a dataset. Use transform when you want to run the stage with a already present/fitted configuration.')
flags.DEFINE_string('config_filename',None,'Filetype json required. Name of the configuration file that will be saved as json when stage_config is fit. When stage_config is transform this should be the name of the configuration file to load.')
flags.DEFINE_string('filename_x',None,'Filename of feature(_x) file for feature selection/extraction')
flags.DEFINE_string('filename_y',None,'Filename of target(_y) file for feature selection')
flags.DEFINE_float('fdr_level',0.05,'The FDR level that should be respected, this is the theoretical expected percentage of irrelevant features among all created features.')
flags.DEFINE_integer('n_jobs',None,'Number of processes to use during the p-value calculation, set 0 to disable Parallelization')
flags.DEFINE_string('chunksize',None,' The size of one chunk that is submitted to the worker process for the parallelisation. The chunksize can have an crucial influence on the optimal cluster performance and should be optimised in benchmarks for the problem at hand.')
flags.DEFINE_string('column_names',"{'column_id':'','column_sort':'', 'column_kind':'', 'column_value':'None'}",'Input Option 2, https://tsfresh.readthedocs.io/en/latest/text/data_formats.html')
flags.DEFINE_string('standard_parameters',None,"['ComprehensiveFCParameters','MinimalFCParameters','EfficientFCParameters'] Select one of three tsfresh default parameter dictionary")
flags.DEFINE_string('fc_parameters',None,'Dict that defines individual function parameters for feature extraction.')
flags.DEFINE_string('kind_parameters',None,'Dict that defines individual kind parameters for feature extraction.')
flags.DEFINE_string('y_target_column',None,'If y shape[1] > 1 --> Column Name of target column which is needed to test which features are relevant. Target Vector can be binary or real-valued.')

flags.mark_flag_as_required('input_path')
flags.mark_flag_as_required('output_path')
flags.mark_flag_as_required('stage')
FLAGS = flags.FLAGS


def main(argv):
    """Runs the possible functions from tsfresh
        1. fit:
            a. extract: Extract Features and saves a configuration dictionary (all columns) afterwards
            b. select: Select Features and saves a configuration dictionary(only relecvant columns) afterwards
            c. both: Extract Features + Select Extracted Features and saves a configuration dictionary(only relecvant columns) afterwards
        2. transform:
            a. extract: Extract Features with configuration from part 1. a.
            b. select: Select Features with a configuration from part 1. b.
            c. both: Exract Features and select extracted features with a configuration from part 1. c.
    Args:
        argv (None): No further arguments should be parsed.
    """    
    del argv 

    if FLAGS.n_jobs == None or FLAGS.n_jobs > os.cpu_count():
        n_jobs = os.cpu_count()
        logging.info('Set n_jobs to maximal number of cpus. Depends to operating system.')
    else:
        n_jobs = FLAGS.n_jobs

    if FLAGS.chunksize:
        chunksize = int(FLAGS.chunksize)
    else:
        chunksize = None

    df_x = read_pd_fth(FLAGS.input_path,FLAGS.filename_x)

    # Config.json(saved to kind_parameters) includes all information we need to transform the df. 
    if FLAGS.stage_config == 'transform':
        kind_parameters = read_json(FLAGS.input_path,FLAGS.config_filename)
        standard_parameters =  None
        fc_parameters = None

        if FLAGS.stage in ['extract','both']:
            column_names = eval(FLAGS.column_names)
            df_x = feature_extraction(df_x=df_x,
                                        col_n=column_names,
                                        standard_parameters=standard_parameters,
                                        n_jobs=n_jobs,
                                        fc_parameters=fc_parameters,
                                        kind_parameters=kind_parameters,
                                        chunksize=chunksize)
            used_settings = feet.settings.from_columns(df_x)
        
        if FLAGS.stage in ['select','both']:
            if FLAGS.stage == 'select':
                df_x = remove_columns(df_x,kind_parameters)
        
    elif FLAGS.stage_config == 'fit':
        standard_parameters = FLAGS.standard_parameters
        kind_parameters = FLAGS.kind_parameters
        fc_parameters = FLAGS.fc_parameters

        if FLAGS.stage in ['extract','both']:
            column_names = eval(FLAGS.column_names)
            df_x = feature_extraction(df_x=df_x,
                                        col_n=column_names,
                                        standard_parameters=standard_parameters,
                                        n_jobs=n_jobs,
                                        fc_parameters=fc_parameters,
                                        kind_parameters=kind_parameters,
                                        chunksize=chunksize)
            used_settings = feet.settings.from_columns(df_x)

        if FLAGS.stage in ['select','both']:
            df_y = read_pd_fth(FLAGS.input_path,FLAGS.filename_y)

            df_x = feature_selection(df_x=df_x,
                        df_y=df_y,
                        fdr_level=FLAGS.fdr_level,
                        n_jobs=n_jobs,
                        chunksize=chunksize,
                        target_col=FLAGS.y_target_column)
        
            if FLAGS.stage == 'select': # Save columnames to config.json
                used_settings = {}
                for col in df_x.columns.to_list():
                    used_settings[col] = ''
            elif FLAGS.stage == 'both': # Save columns from extract and select to config.json
                used_settings = feet.settings.from_columns(df_x)
            
            to_pd_fth(FLAGS.output_path,FLAGS.filename_y,df_y)
        

        to_json(FLAGS.output_path,FLAGS.config_filename,used_settings)

    to_pd_fth(FLAGS.output_path,FLAGS.filename_x,df_x)

    
if __name__ == '__main__':
    app.run(main)