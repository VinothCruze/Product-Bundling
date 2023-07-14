from dataset_creation import dataset_creation
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules
#FPGrowth algorithms
from pyspark.ml.fpm import FPGrowth


# preprocess the data to form the dataset as applicable to ALS
def preprocessing(invoice_na):
    invoice_filter = invoice_na.filter(F.col('customer_area') == 'Metal')
    prod_turnover = invoice_filter.groupby(F.col('customernumber'), F.col('model_number')).agg(F.sum(F.col('sales_value')).alias("turnover"))
    prod_turnover = prod_turnover.na.drop(subset=["model_number"])
    prod_turnover = prod_turnover.where(F.col("turnover") > 0)
    mean_prod_turnover = prod_turnover.groupby(F.col('model_number')).agg(F.mean("turnover").alias("avg_turnover"))
    mean_prod_turnover = mean_prod_turnover.where(F.col("avg_turnover") != 0)
    prod_turnover = prod_turnover.join(mean_prod_turnover, [prod_turnover.model_number == mean_prod_turnover.model_number]).select(prod_turnover["*"],mean_prod_turnover["avg_turnover"])

    # calculate implicit rating
    ncf_invoices = prod_turnover.withColumn("raw_rating", (F.col("turnover") / F.col("avg_turnover")))
    ncf_invoices = ncf_invoices.withColumn("purchased", F.lit(1))

    # Flatten Outlier
    ncf_invoices = ncf_invoices.withColumn("rating", F.when(F.col("raw_rating") > 10, 10).otherwise(
                        ncf_invoices.raw_rating))

    ncf_invoices = VectorAssembler(inputCols=["rating"], outputCol="rating_vector").transform(ncf_invoices)

    scaler = StandardScaler(inputCol="rating_vector", outputCol="std_scaled_rating")
    ncf_invoices = scaler.fit(ncf_invoices).transform(ncf_invoices)

    scaler = MinMaxScaler(min=0.0, max=1.0, inputCol="rating_vector", outputCol="minmax_scaled_rating")
    ncf_invoices = scaler.fit(ncf_invoices).transform(ncf_invoices)
    indexer_customer = StringIndexer(inputCol="customernumber", outputCol="customer_id", handleInvalid="keep")
    indexer_product = StringIndexer(inputCol="model_number", outputCol="product_id", handleInvalid="keep")
    pipeline = Pipeline(stages=[indexer_customer, indexer_product])
    alsmod = pipeline.fit(ncf_invoices).transform(ncf_invoices)
    return ncf_invoices

def ALS(alsmod):
    data = alsmod 
    data = data.repartition(20)
    data = data.cache()
    #
    (train, test) = data.randomSplit([0.7, 0.3], seed=123)
    """parameters: regParam=Regularization parameter, nonnegative = to consider nonnegative values for the ALS,implicitPrefs & alpha=to consider for implicit data,
                coldStartstrategy= for the purpose of test unseen data of interaction avoided coz of inclusion of null/nan values
                rank is the number of latent factors in the model """
    als = ALS(rank=70, maxIter=50, regParam=0.15, alpha=1,
                    userCol="customer_id", itemCol="product_id", ratingCol="minmax_scaled_rating_col",
                    coldStartStrategy="drop",
                    nonnegative=True,
                    implicitPrefs=True,
                    seed=123) 

    
    model = als.fit(train)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="minmax_scaled_rating_col",
                                    predictionCol="prediction")
    return model

def ALS_results(model):
    # to specify number of products to be recommended for each customer
    n_reco_prod = 50 
    user_recos = model.recommendForAllUsers(n_reco_prod)

    reco = user_recos.withColumn("rec_exp", F.explode("recommendations")) \
                .select("customer_id", F.col("rec_exp.product_id"), F.col("rec_exp.rating"))

    # resolve IDs to Customer and Product number
    customer = data.select(invoice_na.customernumber, "customer_id").distinct()
    product = data.select(invoice_na.model_number, "product_id").distinct()

    raw_result = reco
    return raw_results

if __name__ == "__main__":
    data = dataset_creation()
    results = preprocessing(data)
    model = ALS(results)
    final = ALS_results(model)



