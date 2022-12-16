'''
Created on Wed Jan 19 2022
@author:Melisa

Create a set pf functions to access the data for 2p.
'''

import os
import logging
import pandas as pd
import numpy as np
import math


steps = ['cropping','motion_correction','alignment','source_extraction','component_evaluation']

#references_path = '/ceph/imaging1/melisa/photon2_test/photon2_references_list.xlsx'
references_path = '/scratch/melisa/photon2_test/photon2_test_croc.xlsx'
# references_path = '/ceph/imaging1/melisa/photon2_test/photon2_janek.xlsx'
# references_path = '/ceph/imaging1/melisa/photon2_test/photon2_test_croc.xlsx'
# references_path = '/ceph/imaging1/melisa/photon2_test/photon2_reference_list_2022.xlsx'
# references_path = '/ceph/imaging1/melisa/photon2_test/camilo_test.xlsx'
# references_path = '/ceph/imaging1/melisa/photon2_test/uni_vs_bi.xlsx'

data_structure = ['mouse', 'year', 'month', 'date', 'example']

analysis_structure = [f'{step}_v' for step in steps]
multi_index_structure = data_structure + analysis_structure

columns = data_structure + ['raw_output']# for each step, add a 'v' (version), 'parameters', 'output' and 'comments' columns
for step in steps:
    columns += [f'{step}_{idx}' for idx in ['v','parameters','output']]


def open_data_base(path = references_path):
    '''
    Opens the data base with the proper data structure
    :param path: path to the excel file with the data base
    :return: pandas formated and structured data base
    '''
    return pd.read_excel(path,dtype={'data':'str','time':'str'}).set_index(multi_index_structure)

def get_query_from_dict(dictionary):
    query = ''
    for key in dictionary:
        if dictionary[key] == None:
            logging.warning('There is a None in the dictionary. None s are not allowed!')
        if query != '':
            query += ' & '
        query += f'{key} == {dictionary[key]}'
    return query

def select(data_base, mouse = None, year = None, month = None, date = None, example = None, analysis_version = [0,0,0,0,0]):
    '''
    Selects a dataset line
    :param data_base:
    :param mouse:
    :param year:
    :param month:
    :param date:
    :param example:
    :return: selected : data.frame with selected rows that satisfied the selection
    '''

    # Create a dictionary with the criteria to select a file (this is data criteria only, not analysis)
    data_criteria_0 = [mouse, year, month, date, example]
    data_criteria = {data_structure[i]: data_criteria_0[i] for i in range(0, len(data_structure)) if
                     data_criteria_0[i] != None}

    query = get_query_from_dict(data_criteria)
    if query != '':
        logging.debug('Selecting rows corresponding to specified data')
        logging.debug('query: ' + query)
        selected = data_base.query(query)
        logging.debug(f'{len(selected)} rows found')
    else:
        selected = data_base

    #now select the analysis step
    analysis_criteria_0 = analysis_version
    analysis_criteria = {analysis_structure[i]: analysis_criteria_0[i] for i in
                         range(0, len(analysis_structure)) if analysis_criteria_0[i] != None}
    query = get_query_from_dict(analysis_criteria)
    if query != '':
        logging.debug('Selecting rows corresponding to specified data')
        logging.debug('query: ' + query)
        selected = selected.query(query)
        logging.debug(f'{len(selected)} rows found')

    # If no data founf
    if selected.empty:
        logging.warning(f'No rows were found for the specified parameters.')

    return selected

def update_data_base(states_df, inp):
    '''
    Update the data base by adding new analysis stages
    '''

    if str(type(inp)) == "<class 'pandas.core.frame.DataFrame'>":
        # If a dataframe is inserted, apply the function recursively
        for index, row in inp.iterrows():
            states_df = update_data_base(states_df, row)
    else:
        # If a row is inserted
        if inp.name in states_df.index:
            # Replace the row in the analysis states dataframe
            logging.debug(f'Replacing row {inp.name} in analysis states dataframe')
            for item, value in inp.iteritems():
                states_df.loc[inp.name, item] = value
        else:
            logging.debug(f'Appending row {inp.name} to analysis states dataframe')
            # Append it to the analysis states dataframe
            states_df = states_df.append(inp)

    return states_df

def replace_at_index1(tup, ix, val):
    lst = list(tup)
    lst[ix] = val
    return tuple(lst)

def dict_compare(d1, d2):
    '''
    This function compares two dictionaries
    :param d1: first dictionary
    :param d2: second dictionary
    :return:
    '''
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same

def modify_data_base_row_name(row,states_db,analysis_step):
    '''
    Take one entry from the data base and update the analysis status
    :param data_entry:
    :param analysis_step:
    :return:
    '''

    analysis_step_name = steps[analysis_step]
    row_local = row.copy()
    name = row_local.name
    entry_criteria = list(name)
    if entry_criteria[analysis_step + 5] != 0:
        entry_query = get_query_from_dict({multi_index_structure[i]: entry_criteria[i]
                                           for i in range(5+analysis_step-1) if entry_criteria!=None})

        if entry_query != '':
            logging.debug('Selecting rows corresponding to specified data')
            logging.debug('query: ' +entry_query)
            data_base_selection = states_db.query(entry_query)
            logging.debug(f'{len(data_base_selection)} rows found')


        max_versions = len(data_base_selection)
        verified_parameters = 0
        if max_versions == 1:
            verified_parameters = max_versions - 1
        else:
            for ii in range(1, max_versions):
                version = data_base_selection.iloc[ii]
                print(version[f'{analysis_step_name}' + '_parameters'])
                a, b, c, d = dict_compare(eval(version[f'{analysis_step_name}' + '_parameters']),
                                                eval(row_local[f'{analysis_step_name}' + '_parameters']))
                if bool(c) or b:
                    verified_parameters = verified_parameters +1
                else:
                    new_name = version.name

        if verified_parameters == max_versions-1:
            new_name = replace_at_index1(name, 5 + analysis_step, 1)

        if verified_parameters == max_versions:
            new_name = replace_at_index1(name, 5 + analysis_step, max_versions)
    else:
        new_name = replace_at_index1(name, 5 + analysis_step, 1)

    row_local.name = new_name
    print(new_name)
    return row_local

def save_analysis_states_database(data_base, path = references_path):
    '''
    This function writes the analysis states dataframe (states_df)
    to the analysis states database (.xlsx file).
    '''
    data_base.reset_index().sort_values(by=multi_index_structure)[columns].to_excel(path, index=False)

    return


def create_file_name(step, name):
    '''
    This function returns a correct basename used for files
    (str, e.g. "mouse_56166_session_2_trial_1_R_v1.3.1")
    given an analysis state index and a step_index
    '''

    entry_string = f"mouse_{name[0]}_year_{name[1]}_month_{name[2]}_day_{name[3]}_example_{name[4]}_"
    analysis_version_string = 'v'
    for i in range(0, step ):
        analysis_version_string += '.'
        analysis_version_string += str(name[5 + i])
    filename = f'{entry_string}_{analysis_version_string}'

    return filename

def create_file_path(subdirectory, name, analysis_step, extension):

    analysis_step_name = steps[analysis_step-1]
    directory = f'data_processing/{analysis_step_name}/' + subdirectory
    fname = create_file_name(analysis_step, name) + extension

    return directory + fname
