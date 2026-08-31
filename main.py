from preprocessing import Read_data_file
from preprocessing import Drop_unnecessary_features
from preprocessing import Check_data_type
from Config import COLS_TO_DROP
df = Read_data_file("Titanic.csv")
if df is not None:
    print(df.head())
    df = Drop_unnecessary_features(df, COLS_TO_DROP)
    print(df.head())
    print(Check_data_type(df))