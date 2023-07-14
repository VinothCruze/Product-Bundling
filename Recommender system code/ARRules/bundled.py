# Import configuration parameters
from config import log_file_path, n_partitions, dataset_file_path, current_date_str, past_date_str, customer_areas
from cust_top10 import customer_top
from fqmprocess import fqmprocessing
from itemsim import item_sim
import pandas as pd
import logging
# Spark initialization
from bdr_tools.wci.spectrumconductor.spark_utils import Spark

app_name = f"bundle"
spark, sc = Spark().get_spark('app_name')
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.shuffle.partitions", n_partitions)
from pyspark.sql import functions as F


# Configure logging
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, filemode='w')




def prod_bundle():
    try:
       # Read the parquet file of the dataset
        invoice_final = spark.read.format("parquet").load(dataset_file_path)
        logging.error("Problem with the invoice dataframe import, check the file read")
        # filter for the invoice only with last 12 months i.e., 01.2022-01.2023
        invoice_filter = invoice_final.filter((F.col('calendar_day') >= current_date_str) & (F.col('calendar_day') <= past_date_str))

        # for the purpose of getting the product name for the model name removing the actual model values with : from the source
        def colon(x):
            sep = ':'
            x = x.split(sep, 1)[0]
            return x

        # for getting the product details dataframe
        prod_list = invoice_filter.select(F.col('model_number'), F.col('product_name'), F.col('product_family'),
                                          F.col('product_class'), F.col('product_group')
                                          , F.col('product_field'))
        prod_listpd = prod_list.toPandas()
        prod_listpd['product_name'] = prod_listpd['product_name'].apply(colon)  # apply colon removal function from description
        logging.info("Product List dataframe is created successfully")
        prod_listpd.drop_duplicates(inplace=True)

        # customer_division = ['ConstrSite project']
        for i in range(0, len(customer_areas)):
            customer_area =  customer_areas[i]
            print(customer_area)
            prod_ds, top10_df = customer_top(customer_area, invoice_filter)
            if not(prod_ds) or not(top10_df):
                logging.error("Problem with the customer_top function check the function log")
                raise SystemExit(1)  # Stop the program execution
            df_fqm, ar_bundle = fqmprocessing(customer_area, top10_df, invoice_filter, prod_listpd)
            if not (df_fqm) or not (ar_bundle):
                logging.error("Problem with the fqmprocessing function check the function log")
                raise SystemExit(1)  # Stop the program execution
            df_itemsim = item_sim(prod_ds, df_fqm, prod_listpd)
            if not (df_itemsim):
                logging.error("Problem with the item_sim function check the function log")
                raise SystemExit(1)  # Stop the program execution
                        ar_bundle.rename(columns={'highest_similarity': 'actual_products', 'consequent': 'most_similar_items',
                                      'rules': 'recommend', 'confidence': 'sim_score'}, inplace=True)
            ar_bundle = ar_bundle[['customernumber', 'actual_products', 'most_similar_items', 'recommend', 'sim_score']]
            df_itemsim = df_itemsim[
                ['customernumber', 'actual_products', 'most_similar_items', 'recommend', 'sim_score']]
            result = pd.concat([df_itemsim, ar_bundle], ignore_index=True, sort=False)
            logging.info("Associative rules and item item similarity based bundles are successfully generated")
        return result
    except Exception as e:
        logging.error('Error occured in bundled function: {}'.format(e))
        return e






