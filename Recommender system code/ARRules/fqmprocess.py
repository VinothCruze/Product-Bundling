from bdr_tools.wci.spectrumconductor.spark_utils import Spark
from config import log_file_path, n_partitions, minSupport, minConfidence, lift
app_name = f"fqm"
spark, sc = Spark().get_spark('app_name')
spark.sparkContext.setLogLevel("ERROR")

spark.conf.set("spark.sql.shuffle.partitions", n_partitions)


# import libraries

from pyspark.sql import functions as F
import warnings
from pyspark.sql.window import Window
warnings.filterwarnings("ignore", category=DeprecationWarning)
# FPGrowth algorithms
from pyspark.ml.fpm import FPGrowth
import logging
import numpy as np
# Configure logging
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, filemode='w')

# for the purpose of Frequent itemset mining
def fqmprocessing(customer_area, top10_df, invoice_filter, prod_listpd):
    """Function to filter the existing dataframe and creating with the required datacolumns.

        Args:
            customer_area: customer divison parameter with values AUTO,CARGO,WOOD,etc
            top10_df     : top10 products for each customer ranked and sorted as derived from previous function
                           customer_top(cust_area).
        Returns:
                 top10_df: dataframe with support, confidence, lift and the product rules with antecedent and consequent
                 ar_bundle: associative rule formed with bundle of products

        """
    # Frequent Itemset Mining
    try:
        # filter with customer area invoice data
        invoice_constr = invoice_filter.filter(F.col('customer_area') == customer_area)
        # remove non bundlable products
        if customer_area == 'Metal':
            invoice_constr = invoice_constr.filter(~F.col('product_family').like('35111102'))

            invoice_constr = invoice_constr.filter(~F.col('product_family').like('35111122'))
        w = Window.partitionBy('customernumber', 'sales_receipt')
        constr = invoice_constr.select(F.col('customernumber'), F.col('sales_receipt'),
                                       (F.array_sort(F.collect_set(F.col('model_number')).over(w)).alias(
                                           'model_number')),
                                       F.size(F.collect_set(F.col('model_number')).over(w)).alias('count'))
        constr = constr.distinct()
        constr_model_number = constr
        logging.info("Frequent Itemset mining algorithm started")
        fpGrowth = FPGrowth(itemsCol="model_number", minSupport=minSupport, minConfidence=minConfidence)
        # Creating the FQM model
        model = fpGrowth.fit(constr_model_number)
        # Apply the filter condition to remove frequent items
        model = model.freqItemsets.filter(F.col('freq') <= 2000)
        AR = model.associationRules
        # Apply the filter condition to remove lift values less than one
        AR = AR.filter((F.col('lift') >= lift))
        # convert the column to rules and then sort it down to have list of items to be there as a bundle
        AR = AR.select(F.col('antecedent'), F.col('consequent'), F.col('confidence'),
                       (F.concat_ws(',', F.col('antecedent')
                                    , F.col('consequent')).alias('rules')))
        AR = AR.cache()
        # Load the association rules data into a pandas DataFrame item_recommend
        AR = AR.withColumnRenamed("sort_array(antecedent, false)", "antecedent")
        df2 = AR.withColumn("antecedent_split", F.explode(AR.antecedent))
        df2_repar = df2.coalesce(1)
        df2_repar = df2_repar.cache()
        data_ar = df2_repar.toPandas()
        # for creating the dictionary for referring the product and its name  == check for the usage, check for toDict() usage
        md = prod_listpd[["model_number", "product_name"]]
        md = md.drop_duplicates()
        md['model_number'] = md['model_number'].astype(int)
        model_dict = {}
        for index, row in md.iterrows():
            key = row['model_number']
            value = row['product_name']
            model_dict[key] = value
        data_ar['consequentitem'] = data_ar['consequent'].str.get(0)

        # similarity dataframe
        da = data_ar[['antecedent', 'consequent', 'confidence']]
        da = da[['antecedent', 'consequent', 'confidence']].loc[
            da[['antecedent', 'consequent', 'confidence']].astype(str).drop_duplicates().index]
        da.reset_index(drop=True, inplace=True)
        logging.info("AR rules generated")
        # for calculating the similarity of the items present in actual and associative rules of antecedant
        def get_highest_similarity(row):
            """calculate similarity of which antecedent items are present in the consequent items.

            Args:
                row: to check on particular row with antecedent and consequent

            Returns:
                antecedent,consequent, confidence of high similarity occurence
            """
            set_1 = set(row['actual_products'])
            try:
                similarities = []
                for index, row2 in da.iterrows():
                    set_2 = set(row2['antecedent'])
                    try:
                        # calculate jaccard similarity
                        similarity = len(set_1.intersection(set_2)) / len(set_1.union(set_2))
                    except:
                        logging.error(set_1, set_2)
                    similarities.append(similarity)
                if max(similarities) == 0:
                    return 0, 0, 0
                else:
                    highest_similarity_index = np.argmax(similarities)
                    # print(highest_similarity_index)
                    return da.loc[highest_similarity_index, 'antecedent'], da.loc[
                        highest_similarity_index, 'consequent'], da.loc[highest_similarity_index, 'confidence']
            except:
                logging.error("Error in the get_highest_similarity function")

        # Calculate the highest similarity for each row in df1
        top10_df[['highest_similarity', 'consequent', 'confidence']] = top10_df.apply(get_highest_similarity, axis=1,
                                                                                      result_type='expand')
        ar_bundle = top10_df[['customernumber', 'highest_similarity', 'consequent', 'confidence']]
        ar_bundle['rules'] = ar_bundle['highest_similarity'] + ar_bundle['consequent']
        ar_bundle = ar_bundle[ar_bundle['confidence'] > 0]
        if not (top10_df) or not (ar_bundle):
            logging.error("Problem with the fqmprocessing function check the function log")
            raise SystemExit(1)  # Stop the program execution
        else:
            logging.info("ar_bundle generated")
        return top10_df, ar_bundle


    except Exception as e:
        print(f"Error: {e}")