from sklearn.linear_model import Lasso
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
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
    
    features_to_encode =['MSSubClass','MSZoning','Street','LotShape',
    'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
    'BldgType','HouseStyle','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st',
    'Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
    'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType',
    'GarageYrBlt','GarageFinish','GarageQual','GarageCond','PavedDrive','MoSold','YrSold',
    'SaleType','SaleCondition']
    for feature in features_to_encode:
        new_df = encode_data(x_df, feature)
    x_df = new_df
    y_df = x_df.pop("SalePrice")

    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=1.0, help="")
    parser.add_argument('--max_iter', type=int, default=1000, help="The maximum number of iterations")
    args = parser.parse_args()

    run = Run.get_context()
    workspace = run.experiment.workspace
    run.log("Alpha:", np.float(args.alpha))
    run.log("Maximum Iteration:", np.int(args.max_iter))

    
    #The dataset is registered using Python SDK in the notebook
    dataset_name = 'Housing Dataset'

    # Get a dataset by name
    ds = Dataset.get_by_name(workspace=workspace, name=dataset_name)
    x,y = clean_data(ds)
    # TODO: Split data into train and test sets.

    ### YOUR CODE HERE ###
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=223)

    model = Lasso(alpha=args.alpha, max_iter=args.max_iter).fit(x_train, y_train)
    y_predict = model.predict(x_test)
    #calculate the root mean squared error
    y_actual = y_test.values.flatten().tolist()
    mse = mean_squared_error(y_actual,y_predict)

    run.log("RMSE", np.sqrt(mse))
    #save the best model
    os.makedirs('outputs', exist_ok = True)
    
    import joblib
    joblib.dump(value = model, filename= 'outputs/model.joblib')

if __name__ == '__main__':
    main()




