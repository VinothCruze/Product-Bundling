from bdr_tools.wci.spectrumconductor.spark_utils import Spark
from config import log_file_path, n_partitions, dataset_file_path, current_date_str, past_date_str, customer_areas

app_name = f"cust_top10"
spark, sc = Spark().get_spark('app_name')
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.shuffle.partitions", n_partitions)

# import libraries

from pyspark.sql import functions as F
import warnings
from pyspark.sql.window import Window
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pyspark.ml.feature import StringIndexer, MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.sql import types as T
import logging

# Configure logging
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, filemode='w')


# for retreiving top 10 products per customer per division
def customer_top(cust_area, invoice_filter):
    """Function to filter the existing dataframe and creating with the required datacolumns.

        Args:
            cust_area: customer divison parameter with values AUTO,CARGO,WOOD,etc

        Returns: top_prod: dataframe with interaction rating of customer and user
                 cust_top10: top10 products for each customer ranked and sorted.

        """
    try:
        invoice_cust = invoice_filter.filter((F.col('customer_area') == cust_area))
        if not (invoice_cust):
            logging.error("Problem with the invoice data loading function check the function log")
            raise SystemExit(1)  # Stop the program execution
        # no of times the customer is buying a product average of it for the whole construction set customers
        order_turnover = invoice_cust.groupby(F.col('customernumber'), F.col('model_number')).agg(
            F.sum(F.col('soldnumber')).alias("product_orders"))
        # remove empty products
        order_turnover = order_turnover.na.drop(subset=["model_number"])

        order_turnover = order_turnover.where(F.col("product_orders") > 0)
        mean_ord_turnover = order_turnover.groupby(F.col('model_number')).agg(F.sum("product_orders").alias("avg_order"))
        mean_ord_turnover = mean_ord_turnover.where(F.col("avg_order") != 0)
        order_turnover = order_turnover.join(mean_ord_turnover,
                                             [order_turnover.model_number == mean_ord_turnover.model_number]) \
            .select(order_turnover["*"], mean_ord_turnover["avg_order"])

        # calculate implicit rating
        ncf_invoices = order_turnover.withColumn("raw_rating", (F.col("product_orders") / F.col("avg_order")))

        # Flatten Outlier
        ncf_invoices = ncf_invoices.withColumn("rating", F.when(F.col("raw_rating") > 10, 10).otherwise(
            ncf_invoices.raw_rating))

        # for the purpose of applying normalizer
        ncf_invoices = VectorAssembler(inputCols=["rating"], outputCol="rating_vector").transform(ncf_invoices)

        # minmax normalization method
        scaler = MinMaxScaler(min=0.0, max=10.0, inputCol="rating_vector", outputCol="minmax_scaled_rating")
        ncf_invoices = scaler.fit(ncf_invoices).transform(ncf_invoices)

        # unlist the vector column
        unlist = F.udf(lambda x: float(list(x)[0]), T.DoubleType())

        top_prod = ncf_invoices.withColumn("interaction", F.round(unlist("minmax_scaled_rating"), 5))
        windowcust = Window.partitionBy("customernumber").orderBy(F.col("interaction").desc())
        top_prod10 = top_prod.withColumn("row", F.row_number().over(windowcust))
        top_prod10 = top_prod10.where(F.col('row').between(1, 10))  # check for the change in size
        top_prod10 = top_prod10.dropna()
        cust_top10 = top_prod10.groupBy("customernumber").agg((F.collect_set("model_number")).alias("Top10_products"))
        cust_top10 = cust_top10.toPandas()
        cust_top10.rename(columns={'Top10_products': 'actual_products'}, inplace=True)
        if not (top_prod) or not (cust_top10):
            logging.error("Problem with the customer_top function check the function log")
            raise SystemExit(1)  # Stop the program execution
        return top_prod, cust_top10
    except Exception as e:
        logging.error('Error occured in customer_top10 function: {}'.format(e))
        return e