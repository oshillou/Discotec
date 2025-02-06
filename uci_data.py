from sklearn import datasets, preprocessing
import pandas as pd
from sklearn import preprocessing
from io import BytesIO
from zipfile import ZipFile
from urllib import request
from ucimlrepo import fetch_ucirepo 

def load_dermatology():
    # fetch dataset 
    dermatology = fetch_ucirepo(id=33) 
    
    # data (as pandas dataframes) 
    X = dermatology.data.features 
    y = dermatology.data.targets 


    X = preprocessing.StandardScaler().fit_transform(X.dropna(axis=1))
    y = y.to_numpy().reshape(-1)-1

    return X, y


def load_digits():
    X, y = datasets.load_digits(return_X_y=True)

    X_preprocessed = preprocessing.StandardScaler().fit_transform(X)

    return X_preprocessed, y

def load_glass():
    response = request.urlopen("https://archive.ics.uci.edu/static/public/42/glass+identification.zip")
    glass_zip = ZipFile(BytesIO(response.read()))

    with glass_zip.open("glass.data") as file:
        data_csv = pd.read_csv(file, index_col=0, header=None)
    
    targets = data_csv[data_csv.columns[-1]]


    # We need to adjust for this dataset as class "4" does not exist. Therefore we have 6 classes and not 7.
    targets = targets - (targets>4) -1 # Now in [0,5]
    data_csv.drop(data_csv.columns[-1], axis=1, inplace=True)

    
    X = preprocessing.StandardScaler().fit_transform(data_csv)

    return X, targets.to_numpy()


def load_ionosphere():
    response = request.urlopen("https://archive.ics.uci.edu/static/public/52/ionosphere.zip")
    ionosphere_zip = ZipFile(BytesIO(response.read()))

    with ionosphere_zip.open("ionosphere.data") as file:
        data_csv = pd.read_csv(file, header=None, index_col=False)

    targets = data_csv[data_csv.columns[-1]]
    data_csv.drop(data_csv.columns[-1], axis=1, inplace=True)


    targets = preprocessing.LabelEncoder().fit_transform(targets)
    X = preprocessing.StandardScaler().fit_transform(data_csv)

    return X, targets

def load_iris():
    X, y = datasets.load_iris(return_X_y=True)

    X_preprocessed = preprocessing.StandardScaler().fit_transform(X)

    return X_preprocessed, y

def load_lung():
    # fetch dataset 
    lung_cancer = fetch_ucirepo(id=62) 
    
    # data (as pandas dataframes) 
    X = lung_cancer.data.features 
    y = lung_cancer.data.targets 


    X = preprocessing.StandardScaler().fit_transform(X.dropna(axis=1))
    y = y.to_numpy().reshape(-1)-1

    return X, y

def load_segmentation():
    response = request.urlopen("https://archive.ics.uci.edu/static/public/147/statlog+image+segmentation.zip")
    segmentation_zip = ZipFile(BytesIO(response.read()))

    with segmentation_zip.open("segment.dat") as file:
        data_csv = pd.read_csv(file, header=None, index_col=False, sep=" ")
    
    targets = data_csv[data_csv.columns[-1]]
    data_csv.drop(data_csv.columns[-1], axis=1, inplace=True)

    
    X = preprocessing.StandardScaler().fit_transform(data_csv)

    # Export
    return X, targets.to_numpy()


def load_wdbc():
    X, y = datasets.load_breast_cancer(return_X_y=True)

    X_preprocessed = preprocessing.StandardScaler().fit_transform(X)

    return X_preprocessed, y


def load_wine():
    X, y = datasets.load_wine(return_X_y=True)

    X_preprocessed = preprocessing.StandardScaler().fit_transform(X)

    return X_preprocessed, y
