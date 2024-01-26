# Data Engineering Pipeline Design 

I completed this project with one team member in order to become comfortable with the ETL workflow using PostgreSQL in AWS. I worked on the project for 3 months ending in June 2023. 

## Web scraping
We scraped data on smoking rates from the Wayback Archive, CDC, and Australian Bureau of Statistics websites. This information will later be used to impute the smoking feature in the heart disease dataset, which is filled with missing values. The 3 websites provide various data on smoking rates by age, sex, race, ethnicity, smoking history, year, and other factors. Of these factors, age and sex are the ones provided for each patient in the heart disease dataset, so these were the factors we focused our scraping on. 

We created a pandas dataframe with 3 features: age, sex, and smoking rate. The dataframe has 152 rows (76 female, 76 male). It contains smoking rates for women and men aged 0-75. As in the previous module of the product, females were assigned smoking rates that correspond to their age group, ad males were assigned smoking rates according to the formula: (smoking rate for that age)*([rate in women]/[rate in men]). 

The smoking rate at each age was calculated differently. For children under age 14, we assigned smoking rates of 0 for females and males. The Wayback Archive site provides smoking rates for adolescents in Grades 9-12, which we approximated as ages 14, 15, 16, and 17, respectively. For ages 18-75, we used the age group statistics from the CDC and Australian websites. For each age between 18 and 75, we averaged the two smoking rates reported by each site. It is important to note that the oldest age group is 75+, so a patient older than 75 can be assumed to have a smoking rate the same as the one corresponding to their sex and age 75. With smoking rates calculated using the formulas for each sex and the rates for each age, a user can easily access a smoking rate using a given patient's age and sex. The resulting dataframe looks like this:

          sex  age smoke_rate
      0     0    0        0.0
      1     0    1        0.0
      2     0    2        0.0
      3     0    3        0.0
      4     0    4        0.0
      ..   ..  ...        ...
      147   1   71  11.154455
      148   1   72  11.154455
      149   1   73  11.154455
      150   1   74  11.154455
      151   1   75    2.20495
      
## Orchestrating workflow in airflow
### Setting up airflow in AWS
We set up an airflow in environment in Amazon MWAA called 19-airflow. It is connected to our S3 bucket, mwaa-19, where the heart_disease_long.csv file is located. The requirements.txt file connected to the airflow environment is also located in that bucket, as well as a dags/ folder where our pipeline is located. In order for our DAG to access items in our S3 bucket, we configured permissions in our MWAA airflow environment according to the files put_permissions.json and get_permissions.json. 

In all 3 of the branches, we use the from_table_to_df decorator function to pass dataframes through different tasks of the branch. We use the create_db_connection function to connect to a postgres database in Amazon RDS (decorator takes tables that have been stored in the db and converts them to pandas dataframes for manipulation in the upcoming task function). We used the de300-rds-test database and allowed our dag tasks to access the database by setting up a connection with conn_id postgres_conn_test in the airflow UI. 

### Branch 1: sklearn
The 3 branches start with the drop_tables task that is a Postgres operator dropping tables and creating a default public schema. The add_data_to_table function then reads the raw heart_disease_long.csv file and loads it into a table "heart" in the postgres database. Each branch has diffferent tasks from this point on. 

The raw data is cleaned for downstream sklearn EDA in the clean_data_func (called in clean_sk_data task). This function takes the original data from the "heart" table as a dataframe. We remove the junk at the end of the file by removing all rows with a NaN target value and drop features with more than 10% of their values missing. If a feature has less than 10% of their values missing, we mode impute categorical features and mean impute numerical features. We remove outlier samples that are more than 2 standard deviations away from the mean and target encode categorical features with multiclass labels. 

The transform_data_func (called by normalize_data task) takes the clean dataframe from the previous task, splits it into a training and test set, normalizes the features of the training set, and log transforms the chol and oldpeak features. The transformed training set and test set are stored in the heart_fe_sklearn and heart_test_sklearn tables, respectively. 

Train_and_evaluate takes a dataframe and model type as input, creates a classifier of the model type (Logreg, random forest, or SVM) and trains the model. The branch name is also taken as input for clarity with naming files since the function is also used in branch 3. The training data from the previous function is input and split further into training and validation sets, and the model is trained on the smaller training set and validated with 5-fold cross validation. We make predictions from the model and report evaluation metrics in the -sklearn_model_results.txt files. The LogReg_skl, SVM_skl, and RF_skl tasks each call functions that call train_and_evaluate with dataframes from the heart_fe_sklearn table as input as well as the corresponding model type. 

A DAG containing only the sklearn branch is in sklearn_branch.py.

### Branch 2: spark
Clean_pyspark_func is called by the task clean_spark_data. Features are imputed according to the instructions from Project module 2. The cleaned rdd is turned into a dataframe then put into the heart_clean_spark table via from_table_to_df. The clean data is used next in the fe_spark task, where fe_spark_func splits the clean df into training and test sets and log1p tranforms selected features. We were able to successfully clean the data in spark but could not get the feature engineering to run properly. We describe what the functions are meant to do here. The transformed data is output into the S3 bucket with the file name transformed_spark.csv, and the transformed df and test df are put into the corresponding tables in the db. Similarly to the previous branch, the RF_spark_data and LR_spark_data tasks take the transformed training set as input, split it further into a training and validation set, and train the models on the smaller training set with 5-fold cross validation. Predictions are made from the models on the validation set and evaluation metrics are reported in the -spark_model_results.txt files.

A DAG containing only the spark branch is in spark_branch2.py.

### Branch 3: scraped
Scrape_data_func is called by the get_smoking_data task and scrapes website data as described above. We use the to_sql method to put this data into a table in the db for use later, and the data is also output to a csv file in our S3 buckt (smoking_rates.csv).

We now want to use the smoking rates we scraped to impute 0/1 smoking values for each patient in the heart_clean_sklearn table based on their age and sex. In the merge_smoking_data task, the merged_scraped_func takes the smoke_data_rates and heart_clean_sklearn tables from the db are taken as input dataframes via from_table_to_df. For each patient in the clean heart disease dataframe, their age and sex is looked up in the smoking rates dataframe. The corresponding rate is then compared to a randomly generated number between 0 and 1, and if the rate is higher than the randomly generated number, the patient gets a value of 1 imputed into their "smoke" column. The resulting clean dataframe with the added smoke column is put into the heart_merged_data table for later use in machine learning models. 

We train LogReg and SVM models using the train_and_evaluate function. The inputs are the dataframe heart_merged_data table, the corresponding model type, and branch name "scrape". The evaluation metrics are in the -scrape_model_results.txt files. 

A DAG containing only the scrape branch is in scrape_branch.py.

## Smoking data in airflow
See info in Branch 3 section. 

Our full dag is in full_dag.py and can be seen in full_dag.png. 




