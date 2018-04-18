"""
* Copyright (C) 2018 Omkar Dhariya - All Rights Reserved
* You may use, distribute and modify this code under the
* terms of the license.
* You should have received a copy of the license with
* this file. If not, please write to: omkar.dhariya@gmail.com
"""

# !/usr/bin/python3

import numpy as np
import pandas as pd
import traceback
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing


########################################################################################
# File Load
def load_csv(file_name, header_index=-1):
    """
    :param file_name: CSV file to read
    :param header_index: Header index if Header present else -1
    :return: Pandas Dataframe
    """
    try:
        if header_index == -1:
            # print("without header ")
            data_frame = pd.read_csv(file_name, header=None)
        else:
            # print("with header ")
            data_frame = pd.read_csv(file_name, header=header_index)
        return data_frame
    except:
        print("Error in the load csv")
        traceback.print_exc()
        return -1


########################################################################################
# Fill Empty
def fill_empty_categorical(data_frame, fill_empty_string="unknown", fill_empty_categorical_type=0):
    """
    :param data_frame: Single column dataframe
    :param fill_empty_string: String to replace
    :param fill_empty_categorical_type: Type of fillna
    :return: df is pandas dataframe
    """
    update_data_frame = data_frame
    try:
        if fill_empty_categorical_type == 0:
            update_data_frame = update_data_frame.fillna(fill_empty_string)
        elif fill_empty_categorical_type == 1:
            update_data_frame = update_data_frame.fillna(update_data_frame.mode()[0])
        else:
            print("Type not found")
        return update_data_frame
    except:
        print("Error in fillna")
        traceback.print_exc()
        return data_frame


def fill_empty_continuous(data_frame, fill_empty_continuous_value=0, fill_empty_continuous_type=0):
    """
    :param data_frame: Input pandas dataframe
    :param fill_empty_continuous_value: Value to replace
    :param fill_empty_continuous_type: Type for fillna 0: with given value 1:With mode 2:With Mean
    :return:
    """

    updated_data_frame = data_frame
    try:
        if fill_empty_continuous_type == 0:
            updated_data_frame = updated_data_frame.fillna(fill_empty_continuous_value)
        elif fill_empty_continuous_type == 1:
            updated_data_frame = updated_data_frame.fillna(updated_data_frame.mode()[0])
        elif fill_empty_continuous_type == 2:
            updated_data_frame = updated_data_frame.fillna(updated_data_frame.mean())
        else:
            print("Type not found hence replacing with default value 0")
            updated_data_frame = updated_data_frame.fillna(fill_empty_continuous_value)
        return updated_data_frame
    except:
        print("Error in fillna")
        traceback.print_exc()
        return data_frame


def fill_empty(data_frame, fill_empty_string="unknown", fill_empty_categorical_type=0,
               fill_empty_continuous_value=0, fill_empty_continuous_type=0):
    """
    :param data_frame: Pandas dataframe
    :param fill_empty_string: String to replace all "Empty or null string with given input"
    :param fill_empty_categorical_type: Type of the fillna i.e 0: with given value 1:With mode
    :param fill_empty_continuous_value: Replace all Empty or NaN or N/A value with given input
    :param fill_empty_continuous_type: Type of the fillna i.e 0: with given value 1:With mode 2:With Mean
    :return: Pandas Dataframe without any empty value
    """
    updated_data_frame = data_frame
    try:
        if len(updated_data_frame) > 0:
            for column in updated_data_frame.columns:
                if data_frame[column].dtype == np.object:
                    updated_data_frame[column] = fill_empty_categorical(data_frame[column], fill_empty_string,
                                                                        fill_empty_categorical_type)
                else:
                    updated_data_frame[column] = fill_empty_continuous(data_frame[column], fill_empty_continuous_value,
                                                                       fill_empty_continuous_type)
        else:
            print("Dataframe is Empty please check read csv")
        return updated_data_frame
    except:
        print("Error in encoding")
        traceback.print_exc()
        return -1


##################################################################################################
# Feature reduction
def remove_unique(data_frame, threshold=90):
    """
    :param data_frame: Pandas Dataframe
    :param threshold: threshold in percent for the removing column by default 90%
    :return: Pandas dataframe
    """
    update_data_frame = data_frame
    try:
        # Converting Threshold to actual count
        threshold_value = len(update_data_frame) * (threshold / 100)
        for column in update_data_frame.columns:
            unique = set(update_data_frame[column])
            if len(unique) > threshold_value:
                # If column have more unique value threshold given
                update_data_frame = update_data_frame.drop(column, axis=1)
            elif len(unique) == 1:
                # If column have only single value
                update_data_frame = update_data_frame.drop(column, axis=1)

        return update_data_frame
    except:
        traceback.print_exc()
        return data_frame


# Removing features with low variance
def remove_low_variance(data_frame, threshold=90):
    """
    :param data_frame: Pandas Dataframe
    :param threshold: threshold in percent for the removing column by default 90%
    :return: Pandas dataframe
    """
    updated_data_frame = data_frame
    try:
        # Converting Threshold to actual count
        threshold_value = (threshold / 100)
        sel = VarianceThreshold(threshold=threshold_value)
        sel.fit(updated_data_frame)
        index = 0
        support_vector = sel.get_support()
        for isSelected in support_vector:
            if not isSelected:
                updated_data_frame = updated_data_frame.drop(updated_data_frame.columns[index], axis=1)
                # With drop index shifted to 1 position left
                index -= 1
            index += 1
        return updated_data_frame
    except:
        traceback.print_exc()
        return data_frame


# Feature importance finding
def find_feature_importance(data_frame, target_column="Null"):
    """
    :param data_frame: Pandas Dataframe
    :param target_column: Target column Header name or null to take last column
    :return: Dictionary of feature importance
    """
    updated_data_frame = data_frame
    try:
        # Dictionary of feature importance
        result = {}
        # Default target column as last column
        target_column_index = len(updated_data_frame.columns) - 1
        # Checking the given column present or not
        if target_column != "Null":
            if target_column in updated_data_frame.columns.tolist():
                target_column_index = updated_data_frame.columns.tolist().index(target_column)
            else:
                print("Target not found")
                return {}

        # Divide data into Value and Target
        data_frame_target = updated_data_frame[updated_data_frame.columns[target_column_index]]
        data_frame_data = updated_data_frame.drop(updated_data_frame.columns[target_column_index], axis=1)
        # fit an Extra Trees model to the data
        model = ExtraTreesClassifier()
        model.fit(data_frame_data.values, data_frame_target)
        # Creating Dictionary the relative importance of each attribute
        index = 0
        for importance in model.feature_importances_:
            result[data_frame_data.columns[index]] = importance * 100
            index += 1
        return result
    except:
        traceback.print_exc()
        return {}


##################################################################################################
# Encoding
# Label Encoder
def label_encoder(data_frame):
    """
    :param data_frame: Pandas Dataframe
    :return: Pandas Dataframe and Encoder dictionary
    """
    encoder_dir = {}
    updated_data_frame = data_frame
    try:
        for column in updated_data_frame.columns:
            if updated_data_frame[column].dtype == np.object:
                le = preprocessing.LabelEncoder()
                updated_data_frame[column] = le.fit_transform(updated_data_frame[column])
                encoder_dir[column] = le
        return updated_data_frame, encoder_dir
    except:
        traceback.print_exc()
        print("Error in Label Encoding")
        return data_frame, encoder_dir


# One Hot Encoder
def one_hot_encoder(data_frame):
    """
    :param data_frame: Pandas Dataframe
    :return: Pandas Dataframe and Encoder dictionary
    """
    encoder_dir = {}
    updated_data_frame = data_frame
    try:
        for column in updated_data_frame.columns:
            if updated_data_frame[column].dtype == np.object:
                dummy_df = pd.get_dummies(updated_data_frame[column])
                dummy_column_list = list()
                for dummy_column in dummy_df.columns:
                    dummy_column_list.append(dummy_column)
                    # print(column)
                    updated_data_frame[str(column) + "__" + str(dummy_column)] = dummy_df[dummy_column]
                updated_data_frame = updated_data_frame.drop(column, axis=1)
                encoder_dir[column] = dummy_column_list

        return updated_data_frame, encoder_dir
    except:
        traceback.print_exc()
        print("Error in One Hot Encoding")
        return data_frame, encoder_dir


# Single value label encoding
def single_value_label_encoding(value, column_name, encoder_dir):
    """
    :param value: Value to encode
    :param column_name: column name which use for encode
    :param encoder_dir: Label encoder trained dictionary
    :return: Encoded value
    """
    try:
        if column_name not in encoder_dir:
            print("Column name not found in encoder dictionary")
            return -1
        label_encoder_object = encoder_dir[column_name]
        value_array = np.asarray(value)
        return label_encoder_object.transform(value_array.ravel())[0]
    except:
        traceback.print_exc()
        print("Error in encoding single value")
        return -1


# Single value one hot encoding
def single_value_one_hot_encoding(value, column_name, encoder_dir):
    """
    :param value: Value to encode
    :param column_name: column name which use for encode
    :param encoder_dir: Label encoder trained dictionary
    :return: Encoded value
    """
    try:
        if column_name not in encoder_dir:
            print("Column name not found in encoder dictionary")
            return -1
        one_hot_encoder_object = encoder_dir[column_name]
        label = list()
        values = list()

        for temp in one_hot_encoder_object:
            label.append(column_name+"__"+temp)
            values.append(0)

        data_frame_list = list()
        data_frame_list.append(values)
        data_frame = pd.DataFrame(data_frame_list, columns=label)
        if value in one_hot_encoder_object:
            data_frame.loc[0, column_name + "__" + value] = 1
        else:
            data_frame.loc[0, column_name + "__unknown"] = 1

        return data_frame
    except:
        traceback.print_exc()
        print("Error in encoding single value")
        return -1


##################################################################################################
# Decoding
# Label Decoder
def label_decoder(data_frame, encoder_dir):
    """
    :param data_frame: Pandas Dataframe
    :param encoder_dir: Directory contains train encoder
    :return: Pandas Dataframe and Encoder dictionary
    """
    updated_data_frame = data_frame
    try:
        for columns in encoder_dir:
            le = encoder_dir[columns]
            updated_data_frame[columns] = le.inverse_transform(updated_data_frame[columns])
        return updated_data_frame
    except:
        traceback.print_exc()
        print("Error in Label Decoding")
        return data_frame


# One Hot Decoder
def one_hot_decoder(data_frame, encoder_dir):
    """
    :param data_frame: Pandas Dataframe
    :param encoder_dir: Directory contains train encoder
    :return: Pandas Dataframe and Encoder dictionary
    """
    updated_data_frame = data_frame
    try:

        for column in encoder_dir:
            updated_data_frame[column] = ""
            for i in range(0, len(updated_data_frame)):
                # label = "unknown"
                for value in encoder_dir[column]:
                    if updated_data_frame[str(column) + "__" + value][i] == 1:
                        updated_data_frame.loc[i, column] = value
            for value in encoder_dir[column]:
                updated_data_frame = updated_data_frame.drop(str(column) + "__" + value, axis=1)

        return updated_data_frame
    except:
        traceback.print_exc()
        print("Error in One Hot Decoding")
        return data_frame, encoder_dir


# Single value label decoding
def single_value_label_decoding(value, column_name, encoder_dir):
    """
    :param value: Value to decode
    :param column_name: column name which use for decode
    :param encoder_dir: Label encoder trained dictionary
    :return: decoded value
    """
    try:
        if column_name not in encoder_dir or value == -1:
            print("Column name not found in encoder dictionary")
            return "unknown"

        label_encoder_object = encoder_dir[column_name]
        value_array = np.asarray(value)
        return label_encoder_object.inverse_transform(value_array.ravel())[0]
    except:
        traceback.print_exc()
        print("Error in decoding single value")
        return "unknown"


# Single value one hot decoding
def single_value_one_hot_decoding(value, column_name, encoder_dir):
    """
    :param value: Value to decode
    :param column_name: column name which use for decode
    :param encoder_dir: Label encoder trained dictionary
    :return: decoded value
    """
    try:
        if column_name not in encoder_dir or value == -1:
            print("Column name not found in encoder dictionary")
            return "unknown"

        one_hot_encoder_object = encoder_dir[column_name]
        temp = value.split("__")
        decoded_value = temp[1]
        if decoded_value in one_hot_encoder_object:
            return decoded_value
        else:
            return "unknown"
    except:
        traceback.print_exc()
        print("Error in decoding single value")
        return "unknown"


##################################################################################################
# Operation on target
# Divide data into data and target
def divide_data_frame(data_frame, target_column="Null"):
    """
    :param data_frame: Input pandas dataframe contains whole CSV
    :param target_column: Target header
    :return:
    """
    try:
        if target_column != "Null":
            data_frame_target = data_frame[target_column]
            data_frame_data = data_frame.drop(target_column, axis=1)
            return data_frame_data, data_frame_target
        else:
            target_column_index = len(data_frame.columns) - 1
            # Divide data into Value and Target
            data_frame_target = data_frame[data_frame.columns[target_column_index]]
            data_frame_data = data_frame.drop(data_frame.columns[target_column_index], axis=1)
            return data_frame_data, data_frame_target
    except:
        traceback.print_exc()
        print("Target Not found")
        return data_frame


# Combine data to One dataframe
def combine_data_frame(data_frame_data, data_frame_target, target_column="Null"):
    """
    :param data_frame_data: Data
    :param data_frame_target: Target Column value
    :param target_column: target column header
    :return:
    """
    try:
        data_frame = data_frame_data
        if target_column != "Null":
            data_frame[data_frame_target] = data_frame_target
        else:
            data_frame[len(data_frame)] = data_frame_target
        return data_frame
    except:
        traceback.print_exc()
        print("Error in combine")
        return data_frame_data

