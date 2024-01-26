from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import boto3
import pandas as pd
# import matplotlib.pyplot as plt
import pathlib

# Pyspark packages
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




def connect_PySpark():
    try:
        spark = SparkSession.builder.appName("Heart").getOrCreate()
        return spark

    except Exception as e:
        print("Spark connection. Error: {}".format(e))
        sys.exit(-1)
    

TABLE_NAMES = {
    "original_data": "heart",
    "smoking_data": "smoke_data_rates",
    "merged_data": "heart_merged_data",
    "clean_sklearn": "heart_clean_sklearn",
    "fe_sklearn": "heart_fe_sklearn",
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
    for ii in range(0, len(df)):
        print(df['age'][ii])
    df.to_sql(TABLE_NAMES['original_data'], conn, if_exists="replace", index=False)

    conn.close()

    return {'status': 1}


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
        #replace_col = df_spark[col_name].astype(float)
        train_df = train_df.withColumn(col_name, np.log1p(df_spark[col_name]))


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
    
    s3_client = boto3.client('s3')
    
    s3_client.upload_file(file_path, 'de300-mwaa-19', 'lr_spark_model_results.txt')

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
        #replace_col = test_data[col_name].astype(float)
        test_data = test_data.withColumn(col_name, np.log1p(test_data[col_name]))

    return test_data


# Instantiate the DAG
dag = DAG(
    'spark_branchv51',
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



#[drop_tables, download_data] >> add_data_to_table >> clean_data >> smoking_data >> merged_data
drop_tables >> add_data_to_table >> clean_spark >> fe_spark >> [spark_RF_data, spark_LR_data]

