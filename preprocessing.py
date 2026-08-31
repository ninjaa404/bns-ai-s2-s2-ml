import pandas as pd
def Read_data_file(file_path):
    try:
        return pd.read_csv(file_path)
    except:
        print("Error reading file")
def Drop_unnecessary_features(df, cols_to_drop):
    df = df.drop(cols_to_drop, axis=1)
    return df
def Check_data_type(df):
    data = pd.DataFrame()
    
    data["Column Name"] = df.columns
    data["Data Type"] = df.dtypes
    data["Unique Values"] = df.nunique()
    
    return data.T