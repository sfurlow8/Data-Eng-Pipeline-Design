from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression as skl_LogReg
#from sklearn.linear_model import LinearRegression  as skl_LinReg
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as skl_RFC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
 
import boto3
import pandas as pd
# import matplotlib.pyplot as plt
import pathlib

# Pyspark packages
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
# from pyspark.ml.clustering import KMeans
from pyspark.mllib.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.classification import LogisticRegression as  spark_LR
from pyspark.ml.classification import RandomForestClassifier as spark_RFC

from pyspark.sql.functions import when


TABLE_NAMES = {
    "original_data": "heart",
    "smoking_data": "smoke_data_rates",
    "merged_data": "heart_merged_data",
    "merged_test_data": "test_merged_data",
    "clean_sklearn": "heart_clean_sklearn",
    "fe_sklearn": "heart_fe_sklearn",
    "logreg_data": "spark_logreg",
    "RF_data": "spark_RF",
    "SVM_data": "spark_SVM",
    "LogReg_scrape_data": "scraped_LogReg",
    "SVM_scrape_data": "scraped_SVM",
    "clean_pyspark": "heart_clean_spark",
    "fe_spark": "heart_fe_spark",
    "skl_test_data": "heart_test_sklearn",
    "spark_test_data": "heart_test_spark",
    "max_fe": "max_fe_features",
    "product_fe": "product_fe_features"
}

ENCODED_SUFFIX = "_encoded"

# Define the default args dictionary for DAG
default_args = {
    'owner': 'sophie',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}

# def read_config() -> dict:
#     path = pathlib.Path(CONFIG_FILE)
#     with path.open(mode="rb") as param_file:
#         params = tomli.load(param_file)
#     return params

# PARAMS = read_config()

def create_db_connection():
    """
    create a db connection to the postgres connection

    return the connection
    """
    
    import re
    from sqlalchemy import create_engine

    conn = BaseHook.get_connection("postgres_conn_test")
    conn_uri = conn.get_uri()

    # replace the driver; airflow connections use postgres which needs to be replaced
    conn_uri= re.sub('^[^:]*://', "postgresql+psycopg2"+'://', conn_uri)

    engine = create_engine(conn_uri)
    conn = engine.connect()

    return conn

def from_table_to_df(input_table_names: list[str], output_table_names: list[str]):
    """
    Decorator to open a list of tables input_table_names, load them in df and pass the dataframe to the function; on exit, it deletes tables in output_table_names
    The function has key = dfs with the value corresponding the list of the dataframes 

    The function must return a dictionary with key dfs; the values must be a list of dictionaries with keys df and table_name; Each df is written to table table_name
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import pandas as pd

            """
            load tables to dataframes
            """
            if input_table_names is None:
                raise ValueError('input_table_names cannot be None')
            
            _input_table_names = None
            if isinstance(input_table_names, str):
                _input_table_names = [input_table_names]
            else:
                _input_table_names = input_table_names

            import pandas as pd
            
            print(f'Loading input tables to dataframes: {_input_table_names}')

            # open the connection
            conn = create_db_connection()

            # read tables and convert to dataframes
            dfs = []
            for table_name in _input_table_names:
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                dfs.append(df)

            if isinstance(input_table_names, str):
                dfs = dfs[0]

            """
            call the main function
            """

            kwargs['dfs'] = dfs
            kwargs['output_table_names'] = output_table_names
            result = func(*args, **kwargs)

            """
            delete tables
            """

            print(f'Deleting tables: {output_table_names}')
            if output_table_names is None:
                _output_table_names = []
            elif isinstance(output_table_names, str):
                _output_table_names = [output_table_names]
            else:
                _output_table_names = output_table_names
            
            print(f"Dropping tables {_output_table_names}")
            for table_name in _output_table_names:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")

            """
            write dataframes in result to tables
            """

            for pairs in result['dfs']:
                df = pairs['df']
                table_name = pairs['table_name']
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"Wrote to table {table_name}")

            conn.close()
            result.pop('dfs')

            return result
        return wrapper
    return decorator

def add_data_to_table_func(**kwargs):
    """
    insert data from local csv to a db table
    """

    import pandas as pd

    conn = create_db_connection()

    df = pd.read_csv("s3://de300-mwaa-19/heart_disease_long.csv", header=0)
    df.to_sql(TABLE_NAMES['original_data'], conn, if_exists="replace", index=False)

    conn.close()

    return {'status': 1}



@from_table_to_df(TABLE_NAMES['original_data'], None)
def clean_data_func(**kwargs):
    """
    data cleaning: drop none, remove outliers based on z-scores
    apply label encoding on categorical variables: assumption is that every string column is categorical
    """
    df = kwargs['dfs']
    # for ii in range(0, len(df)):
    #     print(df['age'][ii])
    
    # missing_value_threshold = 0.8


    # remove junk at the end of the file

    df = df.rename(columns={'ekgday(day': "ekgday"})
    df = df[df['target'].notna()]
    # df = df.iloc[:899]
    #df = df.head(899)

    df = df.rename(columns={'ekgday(day': "ekgday"})
    # for ii in range(0, len(df)):
    #     print(df['age'][ii])

    #### Analyze missing values #####

    # calculate percent of missing values in each feature
    # if a feature has more than 10% missing values, delete it, otherwise mean impute
    total_rows = len(df)
    # print(total_rows)
    clean_df = df
    # print(clean_df)
    # print(len(clean_df))

    clean_df = clean_df.replace([''], np.nan)
    clean_df = clean_df.astype(float)

    num_cols = ["age", "trestbps", "chol", "thaldur", "thalach", "thalrest",
                "tpeakbps", "tpeakbpd", "dummy", "trestbpd", "oldpeak"]
    cat_cols = ['sex', 'cp', 'htn', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig',
                'prop', 'nitr', 'pro', 'diuretic', 'exang', 'xhypo', 'cmo',
                'cday', 'cyr']
    
    # for col in num_cols:
    #     print(clean_df[col])
    #     clean_df[col] = clean_df[col].astype(float)

    for col in clean_df.columns:
        missing = df[col].isna()
        num_missing = sum(missing)
        pct_missing = num_missing/total_rows # percent of the column that is missing values
        
        if pct_missing > 0:
            #print(col, ": ", round(pct_missing*100,2), "%", "missing")
        
            # if more than 10% of the values in the column are missing, delete the column
            if pct_missing > 0.1:
                clean_df = clean_df.drop(col, axis=1)
                #print('dropping column')

            # if less than 10% of the values in the column are missing, mean impute the missing values
            if pct_missing < 0.1:
                if col in num_cols:
                    #print("Mean imputing values for", col, "...")
                    avg = clean_df[col].mean()
                    clean_df.loc[missing, col] = avg
                    #print("Done", "\n")
                else:
                    #print("Mode imputing values for", col, "...")
                    mode_value = clean_df[col].mode()[0]
                    clean_df.loc[missing, col] = mode_value

    #print(clean_df) # look at the clean dataframe
    #print(clean_df.columns) # what columns are left


    ##### Remove outliers #####

    # if a sample is more than 2 standard deviations from the mean, remove the outlier
    for col in num_cols:
        col_sd = clean_df[col].std()
        col_mean = clean_df[col].mean()
        
        clean_df = clean_df[
            (clean_df[col] >= col_mean - 2*col_sd) & 
            (clean_df[col] <= col_mean + 2*col_sd)
        ]


    ##### Transformations #####

    skewed_features = []
    skew_cutoff = 0.5

    for col in num_cols:
        col_skew = clean_df[col].skew()
        #print(col, "skewness =", col_skew)
        
        if abs(col_skew) > skew_cutoff:
            skewed_features.append(col) 

    #print(skewed_features)

    # # log transform oldpeak feature
    # feature = "oldpeak"
    # clean_df[feature] = np.log(abs(clean_df[feature])+1)
    # clean_df = clean_df.rename(columns={feature: "log_"+feature})

    # # normalize and log transform cholesterol feature
    # feature = "chol"
    # feature_mean = clean_df[feature].mean()
    # feature_std = clean_df[feature].std()

    # clean_df['chol_z'] = (clean_df[feature]-feature_mean)/(feature_std)
    # clean_df['chol_z_log'] = np.log(abs(clean_df['chol_z'])+1)

    # clean_df = clean_df.drop(['chol', 'chol_z'], axis = 1)


    ### transform categorical vars

    # target encode features with more than 01 categories
    for feature in cat_cols:
        if len(clean_df[feature].unique()) > 2:
            mean_encoding = clean_df.groupby(feature)['target'].transform('mean')
            clean_df[feature+"_target"] = mean_encoding
            clean_df = clean_df.drop(feature, axis = 1)

    num_cols = ["age", "trestbps", "chol_z_log", "thaldur", "thalach", "thalrest",
                "tpeakbps", "tpeakbpd", "dummy", "trestbpd", "log_oldpeak"]

    # fig, axs = plt.subplots(5, 6, figsize=(20,12))

    # ### Histograms for transformed features
    # for col in clean_df.columns:
    #     ind = clean_df.columns.get_loc(col)
    #     axx = int(np.floor(ind/6))
    #     axy = ind%6
    #     axs[axx, axy].set_title(col)
    #     if col in num_cols:
    #         n_bins = 20
    #     else:
    #         n_bins = len(clean_df[col].unique())
    #     axs[axx, axy].hist(clean_df[col], bins=n_bins)

    # fig.tight_layout()

    # ##### Boxplots for features ####
    # df_0 = clean_df[clean_df['target']==0]
    # df_1 = clean_df[clean_df['target']==1]

    # fig, axs = plt.subplots(5, 6, figsize=(20,12))

    # for col in clean_df.columns:
    #     ind = clean_df.columns.get_loc(col)
    #     axx = int(np.floor(ind/6))
    #     axy = ind%6
    #     axs[axx, axy].set_title(col)
    #     axs[axx, axy].boxplot([df_1[col], df_0[col]], labels = ["Target=1", "Target=0"])

    # fig.tight_layout()
    # plt.show()
    
    # plt.savefig('/tmp/skl-featureplots.png')
    # # plots_to_save.append('featureplots.png')
    # plt.clf()

    boto = boto3.session.Session()     
    #s3 = boto.client('s3')
    s3_client = boto.client('s3')

    # s3_client.upload_file("de300-mwaa-19-output", 'skl-featureplots.png', "/tmp/skl-featureplots.png")

    
    file_path = '/tmp/clean_sklearn.csv'
    clean_df.to_csv(file_path, index=False)
    
    s3_client = boto3.client('s3')
    #upload_file(Filename, Bucket, Key, ExtraArgs=None, Callback=None, Config=None)


    s3_client.upload_file(file_path, 'de300-mwaa-19', 'clean_sklearn.csv')

    return {
        'dfs': [
            {'df': clean_df, 
             'table_name': TABLE_NAMES['clean_sklearn']
             }]
        }

def scrape_data_func():
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup

    df = pd.DataFrame(columns=['sex', 'age', 'smoke_rate'])

    ages = []
    for ii in range(0,76):
      ages.append(ii)

    for jj in range(0, 76):
        ages.append(jj)

    df['age'] = ages
    df['sex'][0:76] = 0
    df['sex'][76:] = 1


    url = "https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/latest-release" # site to scrape
    response = requests.get(url).text
    soup = BeautifulSoup(response, "html.parser")

    # get table attributes
    tab = soup.find_all("table")[0]
    #print(tab)
    age_rates = [] # list of lists
    for tr in tab.find_all("tr"): 
      if tr.find("td") is not None: # age rates in tr's
          age_group = tr.find("th").text
          rate = tr.find("td").text
          age_rates.append([age_group, rate])

    # print(age_rates)


    url = "https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm" # site to scrape
    response = requests.get(url).text
    soup = BeautifulSoup(response, "html.parser")

    tab = soup.find_all("ul")[3] # block list of smoking info by age
    age_rates_2 = []
    list_index = 0
    for list_item in tab.find_all("li"): # loop through list of age groups and rates
        body = list_item.text
        if list_index == 0: # first list item (18-24)
            age_group = body[33:38]
            rate = body[46:49]
        elif (list_index == 1) or (list_index == 2): # 2nd and 3rd list item have same words
            age_group = body[35:40]
            rate = body[48:52]
        else: # last list item (65+)
            age_group = f"{body[33:35]}+"
            rate = body[53:56]
        age_rates_2.append([age_group, rate])
        list_index = list_index + 1 # increment to next list item

    # print(age_rates_2)

    tab = soup.find_all("ul")[2] # block list of smoking info by sex
    sex_rates = []
    list_index = 0
    for list_item in tab.find_all("li"):
        body = list_item.text
        sex = body[6:8]
        if list_index == 0:
            sex = body[28:31]
            rate = body[33:37]
        else: 
            sex = body[28:33]
            rate = body[35:39]
        sex_rates.append([sex, rate])
        list_index = list_index + 1


    url = "https://wayback.archive-it.org/5774/20211119125806/https:/www.healthypeople.gov/2020/data-search/Search-the-Data?nid=5342" # site to scrape
    response = requests.get(url).text
    soup = BeautifulSoup(response, "html.parser")

    tab = soup.find_all("div", {"class": "ds-data-table-container"})[0]
    kid_rates = []
    info_2017 = tab.find_all("span", {"class": "ds-data-point ds-2017"})
    for index in range(0, len(info_2017)):
        span = info_2017[index].find_all("span")
        for index2 in range(0, len(span)):
            if span[index2]['class'][0] == "dp-data-estimate":
                if span[index2].text != '':
                    rate = span[index2].text
                    kid_rates.append(rate)

    # print(kid_rates)

    # smoking rates for kids aged 14, 15, 16, 17
    kid_age_rates = kid_rates[-4:]

    # smoking rates for kids sex=M, sex=F
    kid_sex_rates = kid_rates[1:3]

    # print(kid_age_rates)
    # print(kid_sex_rates)

    # df["smoke_rate"][0:14] = 0
    # df["smoke_rate"][14:18] = kid_age_rates

    # print(df[10:20])

    tmp_ages = []
    for ii in range(0, 76):
      tmp_ages.append(ii)

    def smoke_calc(sex, age):
        rate1 = 0
        rate2 = 0
        smoke_rate = 0
        age = int(age)
        if (age >= 14) & (age <= 17):
            index = age-14
            smoke_rate = kid_age_rates[index]
        elif (age >= 18) & (age < 24):
            rate1 = age_rates[0][1]
            rate2 = age_rates_2[0][1]
        elif (age >= 25) & (age < 34):
            rate1 = age_rates[1][1]
            rate2 = age_rates_2[1][1]
        elif (age >= 35) & (age < 44):
            rate1 = age_rates[2][1]
            rate2 = age_rates_2[1][1]
        elif (age >= 45) & (age < 54):
            rate1 = age_rates[3][1]
            rate2 = age_rates_2[2][1]
        elif (age >= 55) & (age < 64):
            rate1 = age_rates[4][1]
            rate2 = age_rates_2[2][1]
        elif (age >= 65) & (age < 74):
            rate1 = age_rates[5][1]
            rate2 = age_rates_2[3][1]
        elif (age >= 75):
            rate1 = age_rates[6][1]
        avg_smoke = (float(rate1) + float(rate2))/2
        
        if sex == 0:
            smoke_rate = avg_smoke
        else:
            men_quotient = float(sex_rates[0][1])/float(sex_rates[1][1])
            smoke_rate = avg_smoke * men_quotient

        return smoke_rate

    for ind in df.index:
        sex = df['sex'][ind]
        age = df['age'][ind]
        print(f"{sex}, {age}")
        df['smoke_rate'][ind] = smoke_calc(sex, age)

    # print(df)

    # return {
    #     'dfs': [
    #         {'df': df.to_json(), 
    #          'table_name': TABLE_NAMES['smoking_data']
    #          }]
    #     }

    conn = create_db_connection()
    df.to_sql(TABLE_NAMES['smoking_data'], conn, if_exists="replace", index=False)
    conn.close()

    import boto3
    boto = boto3.session.Session()     
    #s3 = boto.client('s3')
    s3_client = boto.client('s3')

    # s3_client.upload_file("de300-mwaa-19-output", 'skl-featureplots.png', "/tmp/skl-featureplots.png")

    
    file_path = '/tmp/smoking_rates.csv'
    df.to_csv(file_path, index=False)
    
    s3_client = boto3.client('s3')
    #upload_file(Filename, Bucket, Key, ExtraArgs=None, Callback=None, Config=None)


    s3_client.upload_file(file_path, 'de300-mwaa-19', 'smoking_rates.csv')

    return {'status': 1}

@from_table_to_df([TABLE_NAMES['smoking_data'], TABLE_NAMES['fe_sklearn']], None)
def merge_scraped_func(**kwargs):
    import random
    dfs = kwargs['dfs']
    scrape_df = dfs[0]
    impute_df = dfs[1]
    print(scrape_df)

    impute_df['smoke'] = np.nan

    for ii in range(0, len(impute_df)):
        age = impute_df['age'][ii]
        sex = impute_df['sex'][ii]
        #smoke_rate = scrape_df['age'==age]['smoke_rate']

        #smoke_rate = impute_df.loc[scrape_df['age'] == age, 'smoke']
        location = scrape_df.loc[(scrape_df['age'] == age) & (scrape_df['sex'] == sex)]
        smoke_rate = location['smoke_rate']
        print(smoke_rate)

        compare_rate = random.random() * 100
        if float(smoke_rate) <= compare_rate:
            smoke = 0
        else:
            smoke = 1
        impute_df['smoke'][ii] = smoke

    df_train, df_test = train_test_split(impute_df, test_size=0.2, random_state=42)


    file_path = '/tmp/merged_scraping_training.csv'
    df_train.to_csv(file_path, index=False)
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, 'de300-mwaa-19', 'merged_scraping_training.csv')



    return {
        'dfs': [
            {'df': df_train, 
             'table_name': TABLE_NAMES['merged_data']
             },
            {
                'df': df_test,
                'table_name': TABLE_NAMES['merged_test_data']   
            },
            ]
        }


@from_table_to_df(TABLE_NAMES['clean_sklearn'], None)
def transform_data_func(**kwargs):
    """
    split to train/test
    normalize and transform 
    """
    from sklearn.model_selection import train_test_split

    df = kwargs['dfs']

    # Split the data into training and test sets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    feature = "oldpeak"
    df_train[feature] = np.log(abs(df_train[feature])+1)
    df_train = df_train.rename(columns={feature: "log_"+feature})

    # normalize and log transform cholesterol feature
    feature = "chol"
    feature_mean = df_train[feature].mean()
    feature_std = df_train[feature].std()

    df_train['chol_z'] = (df_train[feature]-feature_mean)/(feature_std)
    df_train['chol_z_log'] = np.log(abs(df_train['chol_z'])+1)

    file_path = '/tmp/transformed_sklearn.csv'
    df_train.to_csv(file_path, index=False)
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, 'de300-mwaa-19', 'transformed_sklearn.csv')

    return {
        'dfs': [
            {
                'df': df_train, 
                'table_name': TABLE_NAMES['fe_sklearn']
            },
            {
                'df': df_test,
                'table_name': TABLE_NAMES['skl_test_data']   
            },
            ]
        }



def train_and_evaluate(data, mdl, branch):
  # should really be train and validate, but I dont want to mess thing up 
    # models = {
    #     'Logistic': LogisticRegression(),
    #     'Linear': LinearRegression(),
    #     'SVM': SVC(),
    #     'RF': RandomForestClassifier()
    # }
    # df = data
    # df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    # xtrain = df_train.drop("target", axis=1)
    # target = df_train["target"]
    # xtest = df_test.drop("target", axis=1)
    # ytest = df_test["target"]
    
    # #name, model = models[mdl]
    # model = models[mdl]
    # scores = cross_val_score(model, xtrain, target, cv=5, scoring='accuracy')
    # model.fit(xtrain, target)
    # y_pred = model.predict(xtest)

    # y_prob = model.predict_proba(xtest)[:, 1]  # Select probabilities for the positive class
    # threshold = 0.5  # Set the threshold for binary classification

    # y_pred = (y_prob > threshold).astype(int)

    # print(ytest)
    # print(y_pred)

    
    # # results = {
    # #     'precision': precision_score(ytest, y_pred),
    # #     'recall': recall_score(ytest, y_pred),
    # #     'f1_score': f1_score(ytest, y_pred)
    # # }

    # results = {
    #         'accuracy': accuracy_score(ytest, y_pred),
    #         'score': model.score(xtest, y_pred)
    # }
    models = {
        'Logistic': skl_LogReg(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=5000),
        'SVM': SVC(),
        'RF': skl_RFC(n_estimators=100, max_depth=5, random_state=0)
    }
    df = data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    xtrain = df_train.drop("target", axis=1)
    target = df_train["target"]
    xtest = df_test.drop("target", axis=1)
    ytest = df_test["target"]
    
    model = models[mdl]
    scores = cross_val_score(model, xtrain, target, cv=5, scoring='accuracy')
    model.fit(xtrain, target)
    y_pred = model.predict(xtest)
    
    results = {
        'precision': precision_score(ytest, y_pred),
        'recall': recall_score(ytest, y_pred),
        'f1_score': f1_score(ytest, y_pred)
    }
    
    branch_name = branch

    import boto3
    boto = boto3.session.Session()     
    #s3 = boto.client('s3')
    s3_client = boto.client('s3')

    file_path = f'/tmp/{mdl}_{branch_name}_model_results.txt'

    save_df = pd.DataFrame.from_dict(results, orient='index')
    save_df.to_csv(file_path, index=False)

    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, 'de300-mwaa-19', f'{mdl}_{branch_name}_model_results.csv')
    return results



@from_table_to_df(TABLE_NAMES['fe_sklearn'], None)
def sklearn_logreg(**kwargs):
    res = train_and_evaluate(kwargs['dfs'], 'Logistic', 'sklearn')
    return {
        'dfs': [
            {
                'df': pd.DataFrame.from_dict(res, orient='index'),
                'table_name': TABLE_NAMES['logreg_data']   
            },
            ]
        }


@from_table_to_df(TABLE_NAMES['fe_sklearn'], None)
def sklearn_SVM(**kwargs):
    res = train_and_evaluate(kwargs['dfs'], 'SVM', 'sklearn')
    return {
        'dfs': [
            {
                'df': pd.DataFrame.from_dict(res, orient='index'),
                'table_name': TABLE_NAMES['SVM_data']   
            },
            ]
        }


@from_table_to_df(TABLE_NAMES['fe_sklearn'], None)
def sklearn_RF(**kwargs):
    res = train_and_evaluate(kwargs['dfs'], 'RF', 'sklearn')
    return {
        'dfs': [
            {
                'df': pd.DataFrame.from_dict(res, orient='index'),
                'table_name': TABLE_NAMES['RF_data']   
            },
            ]
        }

@from_table_to_df(TABLE_NAMES['merged_data'], None)
def scrape_LogReg(**kwargs):
    res = train_and_evaluate(kwargs['dfs'], 'Logistic', 'scrape')
    return {
        'dfs': [
            {
                'df': pd.DataFrame.from_dict(res, orient='index'),
                'table_name': TABLE_NAMES['LogReg_scrape_data']   
            },
            ]
        }

@from_table_to_df(TABLE_NAMES['merged_data'], None)
def scrape_SVM(**kwargs):
    res = train_and_evaluate(kwargs['dfs'], 'SVM', 'scrape')
    return {
        'dfs': [
            {
                'df': pd.DataFrame.from_dict(res, orient='index'),
                'table_name': TABLE_NAMES['SVM_scrape_data']   
            },
            ]
        }


def connect_PySpark():
    import sys
    try:
        spark = SparkSession.builder.appName("Heart").getOrCreate()
        return spark

    except Exception as e:
        print("Spark connection. Error: {}".format(e))
        sys.exit(-1)

    

@from_table_to_df(TABLE_NAMES['original_data'], None)
def clean_pyspark_func(**kwargs):
    """
    data cleaning
    apply label encoding on categorical variables: assumption is that every string column is categorical
    """
    
    df_pd = kwargs['dfs']
    df_pd = df_pd[df_pd['target'].notna()]

    schema = StructType([
        StructField("age", StringType(), True),
        StructField("sex", StringType(), True),
        StructField("painloc", StringType(), True),
        StructField("painexer", StringType(), True),
        StructField("relrest", StringType(), True),
        StructField("pncaden", StringType(), True),
        StructField("cp", StringType(), True),
        StructField("trestbps", StringType(), True),
        StructField("htn", StringType(), True),
        StructField("chol", StringType(), True),
        StructField("smoke", StringType(), True),
        StructField("cigs", StringType(), True),
        StructField("years", StringType(), True),
        StructField("fbs", StringType(), True),
        StructField("dm", StringType(), True),

        StructField("famhist", StringType(), True),
        StructField("restecg", StringType(), True),
        StructField("ekgmo", StringType(), True),
        StructField("ekgday", StringType(), True),
        StructField("ekgyr", StringType(), True),
        StructField("dig", StringType(), True),
        StructField("prop", StringType(), True),
        StructField("nitr", StringType(), True),

        StructField("pro", StringType(), True),
        StructField("diuretic", StringType(), True),
        StructField("proto", StringType(), True),
        StructField("thaldur", StringType(), True),
        StructField("thaltime", StringType(), True),
        StructField("met", StringType(), True),

        StructField("thalach", StringType(), True),
        StructField("thalrest", StringType(), True),
        StructField("tpeakbps", StringType(), True),
        StructField("tpeakbpd", StringType(), True),
        StructField("dummy", StringType(), True),
        StructField("trestbpd", StringType(), True),
        StructField("exang", StringType(), True),
        StructField("xhypo", StringType(), True),
        StructField("oldpeak", StringType(), True),
        StructField("slope", StringType(), True),
        StructField("rldv5", StringType(), True),
        StructField("rldv5e", StringType(), True),
        StructField("ca", StringType(), True),
        StructField("restckm", StringType(), True),

        StructField("exerckm", StringType(), True),
        StructField("restef", StringType(), True),
        StructField("restwm", StringType(), True),
        StructField("exeref", StringType(), True),
        StructField("exerwm", StringType(), True),
        StructField("thal", StringType(), True),
        StructField("thalsev", StringType(), True),
        StructField("thalpul", StringType(), True),
        StructField("earlobe", StringType(), True),
        StructField("cmo", StringType(), True),
        StructField("cday", StringType(), True),
        StructField("cyr", StringType(), True),
        StructField("target", StringType(), True),
    ])

    spark = connect_PySpark()
    df_spark = spark.createDataFrame(df_pd, schema)
    
    df = df_spark.dropna(subset=['sex'])

    print(df)
    
    rdd_data = df.rdd
    print(rdd_data.first())
    # rdd_data = rdd_data.zipWithIndex()

    # rdd_data = rdd_data.filter(lambda x: x[1] <= 899).map(lambda x: x[0])
    # h = rdd_data.first().split(",")
    # # print(type(h))
    # print(h)
    # rdd_data = rdd_data.map(lambda x: tuple(x.split(","))) # each cell between commas

    r = rdd_data.map(lambda x: (x[0], x[1], x[2], x[3], x[6], x[7], x[10], x[13], x[21], x[22],x[23], x[24], x[26], x[29], x[35], x[37], x[38], x[55]))
    #print(r.collect())

    # remove header
    # h = r.first() 
    # print(h)
    # r2 = r.filter(lambda row: row != h)

    # helper function converts string rdd values to floats
    def conv_to_float(col_val):
      if col_val == "":
        return float("nan")
      else:
        return float(col_val)

    # r2 is rdd with all float values, nan's where applicable
    r2 = r.map(lambda x: (conv_to_float(x[0]), conv_to_float(x[1]), conv_to_float(x[2]), conv_to_float(x[3]), 
                                        conv_to_float(x[4]), conv_to_float(x[5]), conv_to_float(x[6]), conv_to_float(x[7]), 
                                        conv_to_float(x[8]), conv_to_float(x[9]), conv_to_float(x[10]), conv_to_float(x[11]), 
                                        conv_to_float(x[12]), conv_to_float(x[13]), conv_to_float(x[14]), conv_to_float(x[15]), conv_to_float(x[16]),
                                        conv_to_float(x[17]))) 
    #col_names = list(h)
    col_names = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 
                 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak',
                'slope', 'target']
    heart_df = r2.toDF(col_names)
    print(heart_df)

    

    # Part a: median impute painloc and painexer
    cols_to_impute = ['painloc', 'painexer']

    names = []
    for col in cols_to_impute:
      names.append(f"imp_{col}") # names of output columns with imputed data

    imputer = Imputer(inputCols=cols_to_impute, outputCols=names)
    imputer.setStrategy("mode")         
    model = imputer.fit(heart_df)
    imputed_df = model.transform(heart_df)

    # Part b: replace values under 100 and missing values
    cols_to_impute = ['trestbps']
    cond = imputed_df.trestbps < 100
    # use condition to impute missing values AND values under 100
    df_to_impute = imputed_df.withColumn(colName = "trestbps", 
                                          col = when(condition = cond, value = float('nan')).otherwise(imputed_df.trestbps))

    names = []
    for col in cols_to_impute:
      names.append(f"imp_{col}")

    imputer = Imputer(inputCols=cols_to_impute, outputCols=names)
    imputer.setStrategy("mean")         
    model = imputer.fit(df_to_impute)
    imputed_df = model.transform(df_to_impute)

    # Part c: replace values less than 0, greater than 4, missing values
    cols_to_impute = ['oldpeak']
    cond = (imputed_df.oldpeak < 0) | (imputed_df.oldpeak > 4)
    df_to_impute = imputed_df.withColumn(colName = "oldpeak", 
                                        col = when(condition = cond, value = float('nan')).otherwise(imputed_df.oldpeak))

    names = []
    for col in cols_to_impute:
      names.append(f"imp_{col}")

    imputer = Imputer(inputCols=cols_to_impute, outputCols=names)
    imputer.setStrategy("mean")         
    model = imputer.fit(df_to_impute)
    imputed_df = model.transform(df_to_impute)

    # Part d: median impute thalach and thaldur
    cols_to_impute = ['thaldur', 'thalach']
    df_to_impute = imputed_df

    names = []
    for col in cols_to_impute:
      names.append(f"imp_{col}")

    imputer = Imputer(inputCols=cols_to_impute, outputCols=names)
    imputer.setStrategy("mean")         
    model = imputer.fit(df_to_impute)
    imputed_df = model.transform(df_to_impute)

    # Part e: replace missing values and values greater than 1
    cols_to_impute = ['fbs', 'prop', 'nitr', 'pro', 'diuretic']
    cond1 = imputed_df.fbs > 1
    cond2 = imputed_df.prop > 1
    cond3 = imputed_df.nitr > 1
    cond4 = imputed_df.pro > 1
    cond5 = imputed_df.diuretic > 1

    df_to_impute = imputed_df.withColumn(colName = "fbs", col = when(condition = cond1, value = float('nan')).otherwise(imputed_df.fbs))\
                            .withColumn(colName = "prop", col = when(condition = cond2, value = float('nan')).otherwise(imputed_df.prop))\
                            .withColumn(colName = "nitr", col = when(condition = cond3, value = float('nan')).otherwise(imputed_df.nitr))\
                            .withColumn(colName = "pro", col = when(condition = cond4, value = float('nan')).otherwise(imputed_df.pro))\
                            .withColumn(colName = "diuretic", col = when(condition = cond5, value = float('nan')).otherwise(imputed_df.diuretic))

    names = []
    for col in cols_to_impute:
      names.append(f"imp_{col}")

    imputer = Imputer(inputCols=cols_to_impute, outputCols=names)
    imputer.setStrategy("mode")         
    model = imputer.fit(df_to_impute)
    imputed_df = model.transform(df_to_impute)

    # Part f: mode impute exang and slope
    cols_to_impute = ['exang', 'slope']
    df_to_impute = imputed_df

    names = []
    for col in cols_to_impute:
      names.append(f"imp_{col}")

    imputer = Imputer(inputCols=cols_to_impute, outputCols=names)
    imputer.setStrategy("mode")         
    model = imputer.fit(df_to_impute)
    imputed_df = model.transform(df_to_impute)

    clean_df = imputed_df.toPandas()

    file_path = '/tmp/clean_spark.csv'
    clean_df.to_csv(file_path, index=False)
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, 'de300-mwaa-19', 'clean_spark.csv')
    

    return {
        'dfs': [
            {'df': clean_df, 
             'table_name': TABLE_NAMES['clean_pyspark']
             }]
        }

@from_table_to_df(TABLE_NAMES['clean_pyspark'], None)
def fe_pyspark_func(**kwargs):
    """
    normalization
    split to train/test
    """
    
    df_pd = kwargs['dfs']
    spark = connect_PySpark()
    df_spark = spark.createDataFrame(df_pd)
    # split train and test
    train_ratio = 0.8  
    test_ratio = 1 - train_ratio
    train_df, test_df = df_spark.randomSplit([train_ratio, test_ratio], seed=42)
    selected_columns = ['imp_trestbps', 'imp_thaldur', 'imp_thalach']
    # Apply log1p transformation to selected columns
    for col_name in selected_columns:
        replace_col = df_spark[col_name].astype(float)
        train_df = train_df.withColumn(col_name, np.log1p(replace_col))


    transformed_df = train_df.toPandas()
    df_test = test_df.toPandas()

    file_path = '/tmp/transformed_spark.csv'
    transformed_df.to_csv(file_path, index=False)
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, 'de300-mwaa-19', 'transformed_spark.csv')


    return {
        'dfs': [
            {
                'df': transformed_df, 
                'table_name': TABLE_NAMES['fe_spark']
            },
            {
                'df': df_test,
                'table_name': TABLE_NAMES['spark_test_data']   
            },
            ]
        }

@from_table_to_df(TABLE_NAMES['fe_spark'], None)
def spark_RF(**kwargs):
  
    df_pd = kwargs['dfs']
    spark = connect_PySpark()
    df_spark = spark.createDataFrame(df_pd)
    train_ratio = 0.8  
    test_ratio = 1 - train_ratio
    train_data, test_data = df_spark.randomSplit([train_ratio, test_ratio], seed=42)

    rf = spark_RFC(labelCol="target")
    
    rf_param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 50, 100]) \
        .addGrid(rf.maxDepth, [5, 10, 20]) \
        .build()
    
    evaluator = BinaryClassificationEvaluator(labelCol="target")    
    rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_param_grid, evaluator=evaluator, numFolds=5)
    rf_model = rf_cv.fit(train_data)
    predictions = rf_model.transform(test_data)

    # Select the prediction and label columns for evaluation
    predictionAndLabels = predictions.select("prediction", "label")

    # Create a MulticlassClassificationEvaluator
    evaluator = MulticlassClassificationEvaluator(metricName="weightedPrecision", labelCol="label")
    # Calculate precision
    precision = evaluator.evaluate(predictionAndLabels)
    # Update the evaluator for recall
    evaluator.setMetricName("weightedRecall")
    recall = evaluator.evaluate(predictionAndLabels)
    # Update the evaluator for F1 score
    evaluator.setMetricName("f1")
    f1_score = evaluator.evaluate(predictionAndLabels)

    results = {}

    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    file_path = '/tmp/rf_spark_model_results.txt'
    f = open(file_path, 'w+')  # open file in append mode
    # f.write("rf results: \n")
    f.write(results['f1_score'])
    f.write(results['recall'])
    f.write(results['precision'])
    f.close()

    save_df = pd.DataFrame.from_dict(results, orient='index')
    save_df.to_csv(file_path, index=False)
    
    # s3_client = boto3.client('s3')
    # s3_client.upload_file(file_path, 'de300-mwaa-19', 'lr_spark_model_results.txt')

    return results

@from_table_to_df(TABLE_NAMES['fe_spark'], None)
def spark_LR(**kwargs):
  
    df_pd = kwargs['dfs']
    spark = connect_PySpark()
    df_spark = spark.createDataFrame(df_pd)
    train_ratio = 0.8  
    test_ratio = 1 - train_ratio
    train_data, test_data = df_spark.randomSplit([train_ratio, test_ratio], seed=42)

    lr = spark_LR(labelCol="target")
    
    lr_param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
        .build()
        
    evaluator = BinaryClassificationEvaluator(labelCol="target")
    lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_param_grid, evaluator=evaluator, numFolds=5)
    lr_model = lr_cv.fit(train_data)
    predictions = lr_model.transform(test_data)

    # Select the prediction and label columns for evaluation
    predictionAndLabels = predictions.select("prediction", "label")

    # Create a MulticlassClassificationEvaluator
    evaluator = MulticlassClassificationEvaluator(metricName="weightedPrecision", labelCol="label")
    # Calculate precision
    precision = evaluator.evaluate(predictionAndLabels)
    # Update the evaluator for recall
    evaluator.setMetricName("weightedRecall")
    recall = evaluator.evaluate(predictionAndLabels)
    # Update the evaluator for F1 score
    evaluator.setMetricName("f1")
    f1_score = evaluator.evaluate(predictionAndLabels)

    results = {}

    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    file_path = '/tmp/lr_spark_model_results.csv'
    # f = open(file_path, 'w+')  # open file in append mode
    # # f.write("lr results: \n")
    # f.write(results['f1_score'])
    # f.close()

    boto = boto3.session.Session()     
    s3_client = boto.client('s3')
    
    
    save_df = pd.DataFrame.from_dict(results, orient='index')
    save_df.to_csv(file_path, index=False)

    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, 'de300-mwaa-19', 'lr_spark_model_results.txt')

    return results


def transform_spark_test(test_data):

    selected_columns = ['imp_trestbps', 'imp_thaldur', 'imp_thalach']
    # Apply log1p transformation to selected columns
    for col_name in selected_columns:
        replace_col = test_data[col_name].astype(float)
        test_data = test_data.withColumn(col_name, np.log1p(replace_col))

    return test_data

def get_best_model(**kwargs):
  
    s3_client = boto3.client('s3')
    f1 = {}
    # Note: need to add the models for the scrapped data stuff later.
    for model in ['Logistic_sklearn', 'SVM_sklearn', 'RF_sklearn', 'Logistic_scrape', 'SVM_scrape', 'rf_spark', 'lr_spark']:
        
        conn = create_db_connection()
        results = pd.read_csv(f"s3://de300-mwaa-19/{model}_model_results.csv", header=0)
        f1[model] = results.iloc[2,0]
        conn.close()

    sorted_f1 = sorted(f1, key=lambda k: f1[k], reverse=True)
    best_model = sorted_f1[0]


    file_path = '/tmp/best_model.txt'
    f = open(file_path, 'w+')  # open file in append mode
    # f.write("lr results: \n")
    f.write(best_model)
    f.close()
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, 'de300-mwaa-19', 'best_model.txt')


def get_tested_spark(modelname, model, train_data, test_data):
    
    spark = connect_PySpark()
    train_data = spark.createDataFrame(train_data)
    test_data = spark.createDataFrame(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="target")    

    if modelname == 'lr_spark':
        lr_param_grid = ParamGridBuilder() \
            .addGrid(model.regParam, [0.01, 0.1, 1.0]) \
            .build()
        cv = CrossValidator(estimator=model, estimatorParamMaps=lr_param_grid, evaluator=evaluator, numFolds=5)
    elif modelname == 'rf_spark':
        rf_param_grid = ParamGridBuilder() \
            .addGrid(model.numTrees, [10, 50, 100]) \
            .addGrid(model.maxDepth, [5, 10, 20]) \
            .build()
        cv = CrossValidator(estimator=model, estimatorParamMaps=rf_param_grid, evaluator=evaluator, numFolds=5)


    trained_model = cv.fit(train_data)

    test_data = transform_spark_test(test_data)

    predictions = trained_model.transform(test_data)
    # Select the prediction and label columns for evaluation
    predictionAndLabels = predictions.select("prediction", "label")

    # Create a MulticlassClassificationEvaluator
    evaluator = MulticlassClassificationEvaluator(metricName="weightedPrecision", labelCol="label")
    # Calculate precision
    precision = evaluator.evaluate(predictionAndLabels)
    # Update the evaluator for recall
    evaluator.setMetricName("weightedRecall")
    recall = evaluator.evaluate(predictionAndLabels)
    # Update the evaluator for F1 score
    evaluator.setMetricName("f1")
    f1_score = evaluator.evaluate(predictionAndLabels)

    
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return results


def get_tested_sklearn(modelname, model, train_data, test_data):
    
    xtrain = train_data.drop("target", axis=1)
    target = train_data["target"]


    feature = "oldpeak"
    test_data[feature] = np.log(abs(test_data[feature])+1)
    test_data = test_data.rename(columns={feature: "log_"+feature})

    feature = "chol"
    feature_mean = test_data[feature].mean()
    feature_std = test_data[feature].std()

    test_data['chol_z'] = (test_data[feature]-feature_mean)/(feature_std)
    test_data['chol_z_log'] = np.log(abs(test_data['chol_z'])+1)

    xtest = test_data.drop("target", axis=1)
    ytest = test_data["target"]
    
    # name, model = models[modelname]
    scores = cross_val_score(model, xtrain, target, cv=5, scoring='accuracy')
    model.fit(xtrain, target)
    y_pred = model.predict(xtest)
    
    results = {
        'precision': precision_score(ytest, y_pred),
        'recall': recall_score(ytest, y_pred),
        'f1_score': f1_score(ytest, y_pred)
    }
    
    return results
    


@from_table_to_df([TABLE_NAMES['fe_spark'], TABLE_NAMES['spark_test_data'], TABLE_NAMES['fe_sklearn'], TABLE_NAMES['skl_test_data'], TABLE_NAMES['merged_data'], TABLE_NAMES['merged_test_data']], None)
def get_tested_model(**kwargs):
    
    # spark = connect_PySpark()

    dfs = kwargs['dfs']
    
    models = {
        'Logistic_sklearn': skl_LogReg(random_state=0, solver='lbfgs', multi_class='ovr'),
        'SVM_sklearn': SVC(),
        'RF_sklearn': skl_RFC(n_estimators=100, max_depth=2, random_state=0),
        'Logistic_scrape': skl_LogReg(random_state=0, solver='lbfgs', multi_class='ovr'),
        'SVM_scrape': SVC(),

        # 'rf_spark': spark_RFC(labelCol="target"),
        # 'lr_spark': spark_LR(labelCol="target"),
    }
    s3_client = boto3.client('s3')
    f1 = {}
    # Note: need to add the models for the scrapped data stuff later.

    file_path = '/tmp/best_model.txt'
    s3_client = boto3.client('s3')
    s3_client.download_file('de300-mwaa-19', 'best_model.txt', '/tmp/best_model.txt')
    
    f = open(file_path, 'r+')  # open file in append mode
    # f.write("lr results: \n")
    modelname = f.read()
    f.close()

    model = models[modelname]
    if "spark" in modelname:
        pass
        train_data = dfs[0]
        test_data = dfs[1]
        results = get_tested_spark(modelname, model, train_data, test_data)
    elif "sklearn" in modelname:
        train_data = dfs[2]
        test_data = dfs[3]
        results = get_tested_sklearn("sklearn", model, train_data, test_data)
    elif "scrape" in modelname:
        train_data = dfs[4]
        test_data = dfs[5]
        results = get_tested_sklearn("scrape", model, train_data, test_data)
    
    
    file_path = '/tmp/best_model_results.txt'

    f = open(file_path, 'w+')  # open file in append mode
    # f.write("lr results: \n")
    f.write('precision: '+ str(results['precision']) +"\n")
    f.write('recall:'+ str(results['recall']) +"\n")
    f.write('f1_score:'+ str(results['f1_score']) +"\n")
    f.close()
    
    s3_client = boto3.client('s3')
    # s3_client.upload_file("de300-mwaa-19", 'best_model_results.txt', file_path)
    s3_client.upload_file(file_path, 'de300-mwaa-19', 'best_model_results.txt')


# Instantiate the DAG
dag = DAG(
    'full_dag',
    default_args=default_args,
    description='Classify with feature engineering and model selection',
    schedule_interval="@daily",
    tags=["de300"]
)

drop_tables = PostgresOperator(
    task_id="drop_tables",
    postgres_conn_id="postgres_conn_test",
    sql=f"""
    DROP SCHEMA public CASCADE;
    CREATE SCHEMA public;
    GRANT ALL ON SCHEMA public TO postgres;
    GRANT ALL ON SCHEMA public TO public;
    COMMENT ON SCHEMA public IS 'standard public schema';
    """,
    dag=dag
)

# download_data = PythonOperator(
#         task_id="download_data",
#         python_callable=download_data_func,
#         provide_context=True,
#         dag=dag
        
# )

add_data_to_table = PythonOperator(
    task_id='add_data_to_table',
    python_callable=add_data_to_table_func,
    provide_context=True,
    dag=dag
)

clean_skl = PythonOperator(
    task_id='clean_sk_data',
    python_callable=clean_data_func,
    provide_context=True,
    dag=dag
)

smoking_data = PythonOperator(
    task_id='get_smoking_data',
    python_callable=scrape_data_func,
    provide_context=True,
    dag=dag
)

merged_data = PythonOperator(
    task_id='merge_smoking_data',
    python_callable=merge_scraped_func,
    provide_context=True,
    dag=dag
)

transform_skl = PythonOperator(
    task_id='normalize_data',
    python_callable=transform_data_func,
    provide_context=True,
    dag=dag
)

LogReg_skl = PythonOperator(
    task_id='LogisticModel_skl',
    python_callable=sklearn_logreg,
    provide_context=True,
    dag=dag
)


SVM_skl = PythonOperator(
    task_id='SVMModel_skl',
    python_callable=sklearn_SVM,
    provide_context=True,
    dag=dag
)

RF_skl = PythonOperator(
    task_id='RFModel_skl',
    python_callable=sklearn_RF,
    provide_context=True,
    dag=dag
)

LogReg_scrape = PythonOperator(
    task_id='LogisticModel_scraped',
    python_callable=scrape_LogReg,
    provide_context=True,
    dag=dag
)


SVM_scrape = PythonOperator(
    task_id='SVMModel_scraped',
    python_callable=scrape_SVM,
    provide_context=True,
    dag=dag
)

clean_spark = PythonOperator(
    task_id='clean_spark_data',
    python_callable=clean_pyspark_func,
    provide_context=True,
    dag=dag
)


fe_spark = PythonOperator(
    task_id='fe_spark',
    python_callable=fe_pyspark_func,
    provide_context=True,
    dag=dag
)

spark_RF_data = PythonOperator(
    task_id='RF_spark_data',
    python_callable=spark_RF,
    provide_context=True,
    dag=dag
)

spark_LR_data = PythonOperator(
    task_id='LR_spark_data',
    python_callable=spark_LR,
    provide_context=True,
    dag=dag
)

best_model = PythonOperator(
    task_id='find_best_model',
    python_callable=get_best_model,
    provide_context=True,
    dag=dag
)

test_model = PythonOperator(
    task_id='test_model',
    python_callable=get_tested_model,
    provide_context=True,
    dag=dag
)






#[drop_tables, download_data] >> add_data_to_table >> clean_data >> smoking_data >> merged_data
drop_tables >> add_data_to_table >> [clean_skl, clean_spark]
clean_skl >> transform_skl
smoking_data >> merged_data >> [LogReg_scrape, SVM_scrape]
transform_skl >> [merged_data, LogReg_skl, SVM_skl, RF_skl]
clean_spark >> fe_spark >> [spark_RF_data, spark_LR_data]

[LogReg_scrape, SVM_scrape, LogReg_skl, SVM_skl, RF_skl, LogReg_scrape, SVM_scrape] >> best_model
best_model >> test_model

