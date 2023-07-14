from bdr_tools.wci.spectrumconductor.spark_utils import Spark

app_name = f"item_sim"
spark, sc = Spark().get_spark('app_name')
spark.sparkContext.setLogLevel("ERROR")
n_partitions = 2
spark.conf.set("spark.sql.shuffle.partitions", n_partitions)


# import libraries

from pyspark.sql import functions as F
import warnings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

# item-item similarity function
def item_sim(prod_ds,
             df_fqm,prod_listpd):
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
                                                                                  c for c in cell if
                                                                                  c not in "'[]").split(
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
