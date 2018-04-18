from preprocessing import *

##################################################################################################
# Driver program
try:
    file_path = "..\data\small.csv"
    # file_path = "D:\study\Study\preprocessing\data\withoutHeader.csv"
    df = load_csv(file_path, header_index=0)
    df = fill_empty(df)
    # print(df)
    # df, encoder_dir_label = label_encoder(df)
    # encoded_value=single_value_label_encoding("D3","Product_Info_2",encoder_dir_label)
    # print(encoded_value)
    # decoded_value = single_value_label_decoding(encoded_value, "Product_Info_2", encoder_dir_label)
    # print(decoded_value)
    df, encoder_dir_one_hot = one_hot_encoder(df)

    # encoded_value=single_value_one_hot_encoding("D3","Product_Info_2",encoder_dir_one_hot)
    # decoded_value = single_value_one_hot_decoding("Product_Info_2__D3", "Product_Info_2", encoder_dir_one_hot)
    # print(decoded_value)
    # df = one_hot_decoder(df, encoder_dir_one_hot)
    # df = label_decoder(df, encoder_dir_label)

    # print(df)

    # df_data, df_target = divide_data_frame(df)
    # print(dfTarget)

    # df_data = remove_unique(df_data,threshold=90)
    # df_data = remove_low_variance(df_data, threshold=90)

    # df = combine_data_frame(df_data, df_target)
    # feature_imp = find_feature_importance(df)
    # print(feature_imp)
except:
    print("Something wrong happened")
    traceback.print_exc()