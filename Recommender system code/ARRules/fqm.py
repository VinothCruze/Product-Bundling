# spark intialization
from bdr_tools.wci.spectrumconductor.spark_utils import Spark

app_name = f"FQM_APP"
spark, sc = Spark().get_spark('app_name')
spark.sparkContext.setLogLevel("ERROR")
n_partitions = 2
spark.conf.set("spark.sql.shuffle.partitions", n_partitions)

# import libraries

from pyspark.sql import functions as F
import warnings
from config import dataset_file_path,file_path
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pyspark.ml.feature import StringIndexer, MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.sql import types as T
from cust_top10 import customer_top
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# FPGrowth algorithms
from pyspark.ml.fpm import FPGrowth

# date time calculation
from datetime import datetime, timedelta

# current date and past 12 months date
current_date = datetime.now().date()
first_date_of_month = current_date.replace(day=1)
date_12_months_back = first_date_of_month.replace(year=first_date_of_month.year - 1)

current_date = first_date_of_month.strftime('%Y-%m-%d')
past_date = date_12_months_back.strftime('%Y-%m-%d')

# read the parquet file of the dataset created with dataset creation module function
invoice_final = spark.read.format("parquet").load(
    "/invoice_finalpc.parquet")
# filter for the invoice only with last 12 months i.e., 01.2022-01.2023
invoice_filter = invoice_final.filter((F.col('calendar_day') >= past_date) & (F.col('calendar_day') <= current_date))


# for the purpose of getting the product name for the model name removing the actual model values with : from the source
def colon(x):
    sep = ':'
    x = x.split(sep, 1)[0]
    return x


# for getting the product details dataframe
prod_list = invoice_filter.select(F.col('model_number'), F.col('product_name'), F.col('product_family'), \
                                  F.col('product_class'), F.col('product_group')
                                  , F.col('product_field'))
prod_listpd = prod_list.toPandas()
prod_listpd['product_name'] = prod_listpd['product_name'].apply(colon)  # apply colon removal function from description
prod_listpd.drop_duplicates(inplace=True)


# for retreiving top 10 products per customer per division
def customer_top(cust_area):
    """Function to filter the existing dataframe and creating with the required datacolumns.

        Args:
            cust_area: customer divison parameter with values AUTO,CARGO,WOOD,etc

        Returns: top_prod: dataframe with interaction rating of customer and user
                 cust_top10: top10 products for each customer ranked and sorted.

        """

    invoice_cust = invoice_filter.filter((F.col('customer_area') == cust_area))
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
    top_prod10 = top_prod10.where(F.col('row').between(1, 10))
    top_prod10 = top_prod10.dropna()
    cust_top10 = top_prod10.groupBy("customernumber").agg((F.collect_set("model_number")).alias("Top10_products"))
    cust_top10 = cust_top10.toPandas()
    cust_top10.rename(columns={'Top10_products': 'actual_products'}, inplace=True)
    return top_prod, cust_top10


# for the purpose of Frequent itemset mining
def fqmprocessing(customer_area, top10_df):
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
        print('entered FQM loop')
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

        fpGrowth = FPGrowth(itemsCol="model_number", minSupport=0.001, minConfidence=0.50)
        # Creating the FQM model
        model = fpGrowth.fit(constr_model_number)
        # Apply the filter condition to remove frequent items
        model = model.freqItemsets.filter(F.col('freq') <= 2000)
        AR = model.associationRules
        # Apply the filter condition to remove lift values less than one
        AR = AR.filter((F.col('lift') >= 1))
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
        # for creating the dictionary for referring the product and its name
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
                        # jaccard similarity
                        similarity = len(set_1.intersection(set_2)) / len(set_1.union(set_2))
                    except:
                        print(set_1, set_2)
                    similarities.append(similarity)
                if max(similarities) == 0:
                    return 0, 0, 0
                else:
                    highest_similarity_index = np.argmax(similarities)
                    # print(highest_similarity_index)
                    return da.loc[highest_similarity_index, 'antecedent'], da.loc[
                        highest_similarity_index, 'consequent'], da.loc[highest_similarity_index, 'confidence']
            except:
                print("Error in the get_highest_similarity function")

        # Calculate the highest similarity for each row in df1
        top10_df[['highest_similarity', 'consequent', 'confidence']] = top10_df.apply(get_highest_similarity, axis=1,
                                                                                      result_type='expand')
        ar_bundle = top10_df[['customernumber', 'highest_similarity', 'consequent', 'confidence']]
        ar_bundle['rules'] = ar_bundle['highest_similarity'] + ar_bundle['consequent']
        ar_bundle = ar_bundle[ar_bundle['confidence'] > 0]
        return top10_df, ar_bundle


    except Exception as e:
        print(f"Error: {e}")


# item-item similarity function
def item_sim(prod_ds,
             df_fqm):
    """calculate item-item similarity for customers who didnt get matched with FQM.

    Args:
        prod_ds: invoice data after interaction score calculation
        df_fqm: result dataframe after frequent itemset mining.

    Returns:
        recommend_nar: item-item similarity results dataframe
    """
    # item item similarity logic
    invoice_itemsim = prod_ds.select(F.col('customernumber'), F.col('model_number'), F.col('interacton_result'))
    invoice_itemsim = invoice_itemsim.coalesce(1)
    item_sim_pd = invoice_itemsim.toPandas()

    customer_item_matrix = item_sim_pd.pivot_table(
        index='customernumber',
        columns='model_number',
        values='interacton_result',
        aggfunc='sum'
    )
    customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)

    item_item_sim_matrix = pd.DataFrame(
        cosine_similarity(customer_item_matrix.T)
    )

    item_item_sim_matrix.columns = customer_item_matrix.T.index

    item_item_sim_matrix['Product_number'] = customer_item_matrix.T.index
    item_item_sim_matrix = item_item_sim_matrix.set_index('Product_number')

    # Find top 2 similar items for each item
    most_similar_users = {}
    sim_scores = {}
    for item in item_item_sim_matrix.index:
        most_similar_users[item] = item_item_sim_matrix.loc[item].nlargest(3)[1:].index.tolist()
        sim_scores[item] = item_item_sim_matrix.loc[item].nlargest(3)[1:].round(3).tolist()
    item_item_sim_matrix['most_similar_items'] = item_item_sim_matrix.index.map(most_similar_users)
    item_item_sim_matrix['sim_score'] = item_item_sim_matrix.index.map(sim_scores)

    # get only neccessary columns
    item_recommend = item_item_sim_matrix.reset_index()
    item_recommend = item_recommend[['Product_number', 'most_similar_items', 'sim_score']]

    # customer who didnt get matched with FQM filtered
    cust_nar = df_fqm[df_fqm['confidence'] == 0]
    cust_nar = cust_nar[['customernumber', 'actual_products']]
    cust_nar['actual_products'] = cust_nar['actual_products'].str[:2]
    cust_item_rec = cust_nar.explode('actual_products').merge(item_recommend, left_on='actual_products',
                                                              right_on='Product_number', how='inner')
    cust_item_rec = cust_item_rec.drop(['Product_number'], axis=1)
    cust_item_rec = cust_item_rec.explode('most_similar_items')
    prod_listpd['model_number'] = prod_listpd['model_number'].astype(str)
    cust_item_rec['most_similar_items'] = cust_item_rec['most_similar_items'].astype(str)
    cust_item_rec = pd.merge(cust_item_rec, prod_listpd, left_on='most_similar_items', right_on='model_number')

    # Drop the duplicate column and rename the remaining column
    cust_item_rec.drop(['model_number', 'product_family', 'product_class', 'product_group', 'product_field'], axis=1,
                       inplace=True)
    cust_item_rec.rename(columns={'product_name': 'similar_product_name'}, inplace=True)

    cust_item_rec['actual_products'] = cust_item_rec['actual_products'].astype(str)

    cust_item_rec = pd.merge(cust_item_rec, prod_listpd, left_on='actual_products', right_on='model_number')

    # Drop the duplicate column and rename the remaining column
    cust_item_rec.drop(['model_number', 'product_family', 'product_class', 'product_group', 'product_field'], axis=1,
                       inplace=True)
    cust_item_rec.rename(columns={'product_name': 'actual_product_name'}, inplace=True)

    # for adding the similarity score
    cust_item = cust_item_rec[['customernumber', 'actual_products', 'sim_score']]
    cust_item = cust_item.explode('sim_score')
    cust_item.drop_duplicates(inplace=True)
    cust_item = cust_item.groupby(['customernumber', 'actual_products']).agg(
        {'sim_score': lambda x: ','.join(set(map(str, x)))})
    cust_item.reset_index(inplace=True)

    # forming the results back to the list
    cust_item_rec = cust_item_rec.groupby(['customernumber', 'actual_products']).agg(
        {'most_similar_items': lambda x: ','.join(set(x)), 'similar_product_name': lambda x: ','.join(set(x)),
         'actual_product_name': lambda x: ','.join(set(x))})

    # to convert the list of values to list

    cust_item_rec.most_similar_items = cust_item_rec.most_similar_items.str.split(',')
    cust_item_rec.similar_product_name = cust_item_rec.similar_product_name.str.split(',')
    cust_item_rec.actual_product_name = cust_item_rec.actual_product_name.str.split(',')

    cust_item_rec.reset_index(inplace=True)

    cust_item_rec_merge = pd.merge(cust_item_rec, cust_item, on=['customernumber', 'actual_products'], how='left')

    # making the list of actual and closeness products based on item-item similarity
    recommend_nar = cust_item_rec_merge
    recommend_nar['actual_products'] = recommend_nar['actual_products'].apply(lambda cell:
                                                                              ''.join(
                                                                                  c for c in cell if c not in "'[]").split(
                                                                                  ', '))
    recommend_nar['recommend'] = recommend_nar['actual_products'] + recommend_nar['most_similar_items']
    # recommend_nar = recommend_nar.drop(['actual_products','most_similar_items'],axis=1)
    recommend_nar['recommend_name'] = recommend_nar['actual_product_name'] + recommend_nar['similar_product_name']
    recommend_nar = recommend_nar.drop(['actual_product_name', 'similar_product_name'], axis=1)
    # to remove the duplicate records
    recommend_nar['recommend'] = recommend_nar.recommend.map(pd.unique)
    recommend_nar['recommend_name'] = recommend_nar.recommend_name.map(pd.unique)
    recommend_nar = recommend_nar.reset_index()
    recommend_nar['rules_count'] = recommend_nar['recommend'].str.len()
    return recommend_nar

if __name__ == "__main__":
    try:
        print("FQM Call")
        customer_division = ['Auto', 'Construction', 'Cargo', 'House Engineering', 'Metal', 'Wood',
                             'ConstrSite project', 'Engineering workshop (Betriebswerkstatt)']
        # customer_division = ['ConstrSite project']
        for i in range(0, len(customer_division)):
            customer_area = customer_division[i]
            print(customer_area)
            prod_ds, top10_df = customer_top(customer_area)
            df_fqm, ar_bundle = fqmprocessing(customer_area, top10_df)
            df_itemsim = item_sim(prod_ds, df_fqm)

            ar_bundle.rename(columns={'highest_similarity': 'actual_products', 'consequent': 'most_similar_items',
                                      'rules': 'recommend', 'confidence': 'sim_score'}, inplace=True)
            ar_bundle = ar_bundle[['customernumber', 'actual_products', 'most_similar_items', 'recommend', 'sim_score']]
            df_itemsim = df_itemsim[
                ['customernumber', 'actual_products', 'most_similar_items', 'recommend', 'sim_score']]
            result = pd.concat([df_itemsim, ar_bundle], ignore_index=True, sort=False)
            # path = '/datalake/WMLA/data/d_1401/projects/d_00209769/code/dev/src/Bundle'
            # pkl = (f'{path}/{customer_area}_merge1.pkl')
            # pd.to_pickle(result, pkl)
            result.write.format("parquet").mode('overwrite').save(f'file_path/{customer_area}')


    except Exception as e:
        print(f"Error: {e}")

