from unittest import TestCase
from preprocessing import *


class Test(TestCase):
    def test_load_csv(self):
        file_name = "test.csv"
        df = load_csv(file_name,header_index=0)

    def test_fill_empty(self):
        file_name = "test.csv"
        df = load_csv(file_name, header_index=0)
        df = fill_empty(df)

    def test_remove_unique(self):
        file_name = "test.csv"
        df = load_csv(file_name, header_index=0)
        df = fill_empty(df)
        df = remove_unique(df, threshold=90)

    def test_remove_low_variance(self):
        file_name = "test.csv"
        df = load_csv(file_name, header_index=0)
        df = fill_empty(df)
        df, encoder_dir_label = label_encoder(df)
        df = remove_low_variance(df, threshold=90)

    def test_find_feature_importance(self):
        file_name = "test.csv"
        df = load_csv(file_name, header_index=0)
        df = fill_empty(df)
        df, encoder_dir_label = label_encoder(df)
        # df = remove_low_variance(df, threshold=90)
        feature_imp = find_feature_importance(df)

    def test_one_hot_encoder(self):
        file_name = "test.csv"
        df = load_csv(file_name, header_index=0)
        df = fill_empty(df)
        df, encoder_dir_label = one_hot_encoder(df)
