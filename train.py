
from sklearn.linear_model import LinearRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory



def clean_data(data):
    
    x_df = data.to_pandas_dataframe()
    x_df = x_df.dropna()

    #clean data
    x_df['MSSubClass'] = x_df['MSSubClass'].astype(str)
    x_df['YrSold'] = x_df['YrSold'].astype(str)
    x_df['MoSold'] = x_df['MoSold'].astype(str)
    x_df['YearBuilt'] = x_df['YearBuilt'].astype(str)
    x_df['YearRemodAdd'] = x_df['YearRemodAdd'].astype(str)
    # One hot encode data   
    #https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python 
    def encode_data(dataframe, feature_to_encode):
        dummies = pd.get_dummies(dataframe[[feature_to_encode]])
        new_df = pd.concat([dataframe,dummies], axis=1)
        new_df = new_df.drop([feature_to_encode], axis=1)
        return(new_df)
    
    features_to_encode =[]
    for feature in features_to_encode:
        new_df = encode_data(x_df, feature)
    x_df = new_df
    y_df = x_df.pop("SalePrice")

    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=int, default=100, help="")
    parser.add_argument('--penalty', type=int, default=2, help="")
    args = parser.parse_args()

    run = Run.get_context()
    workspace = run.experiment.workspace
    run.log("C:", np.int(args.C))
    run.log("Penalty:", np.int(args.penalty))

    
    #The dataset is registered using Python SDK in the notebook
    dataset_name = 'Housing Dataset'

    # Get a dataset by name
    ds = Dataset.get_by_name(workspace=workspace, name=dataset_name)
    
    # TODO: Split data into train and test sets.

    ### YOUR CODE HERE ###
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=223)

    model = LinearRegression(C=args.C, penalty=args.penalty).fit(x_train, y_train)

    #accuracy = model.score(x_test, y_test)

    run.log("Accuracy", np.float(accuracy))
    #save the best model
    os.makedirs('outputs', exist_ok = True)
    
    import joblib
    joblib.dump(value = model, filename= 'outputs/model.joblib')

if __name__ == '__main__':
    main()
© 2021 GitHub, Inc.




