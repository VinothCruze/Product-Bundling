# for NCF implementation
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from tensorflow.keras.layers import (Concatenate, Dense, Dropout, Embedding, Flatten, Input, Multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


# Hyperparameters
latent_dim_mf = 32
latent_dim_mlp = 32
mlp_layer_sizes = [64, 32, 16, 8]
dropout_rates = [0.1, 0.1, 0.1, 0.1]
regs = [0.0001, 0.0001, 0.0001, 0.0001]
activation_fn = 'relu'
lr = 0.0001


def preprocessing(input_data):
    invoice_final = input_data
    #invoice_final = spark.read.format("parquet").load("/datalake/WMLA/data/d_1401/projects/d_00209769/code/dev/src/Bundle/parquet/invoice_final.parquet")
    # filter for the invoice only with last 12 months i.e., 01.2022-01.2023
    invoice_filter = invoice_final.filter((F.col('calendar_day') >= '2022-01-01') & (F.col('calendar_day') <= '2023-02-01'))
    invoice_filter = invoice_filter.withColumn('month_year', F.concat(F.col('sales_year'),F.lit('-'),F.col('sales_month')))
    aggregate = invoice_filter.groupBy(F.col('customernumber')).agg(F.countDistinct('month_year'))
    aggregate = aggregate.filter(F.col('count(month_year)')>5)
    invoice_filter = invoice_filter.join(aggregate, [invoice_filter.customernumber == aggregate.customernumber]).select(invoice_filter["*"],aggregate["count(month_year)"])
    model_aggregation = invoice_filter.groupBy("customernumber").agg(F.size(F.collect_set("model_number"))
                                                        .alias("Distinct_products_count"))
    model_aggregation = model_aggregation.filter(F.col('Distinct_products_count')>5)
    invoice_filter = invoice_filter.join(model_aggregation, [invoice_filter.customernumber == model_aggregation.customernumber]).select(invoice_filter["*"],model_aggregation["Distinct_products_count"])

    invoice_filter = invoice_filter.filter((F.col('customer_area') == customer_area)) 


    # no of times the customer is buying a product average of it for the whole construction set customers
    order_turnover = invoice_filter.groupby(F.col('customernumber'), F.col('model_number')).agg(F.sum(F.col('soldnumber')).alias("product_orders"))
    # remove empty products
    order_turnover = order_turnover.na.drop(subset=["model_number"])

    order_turnover = order_turnover.where(F.col("product_orders") > 0)
    mean_ord_turnover = order_turnover.groupby(F.col('model_number')).agg(F.mean("product_orders").alias("avg_order"))
    mean_ord_turnover = mean_ord_turnover.where(F.col("avg_order") != 0)
    order_turnover = order_turnover.join(mean_ord_turnover, [order_turnover.model_number == mean_ord_turnover.model_number]).select(order_turnover["*"],mean_ord_turnover["avg_order"])

    # calculate implicit rating
    ncf_invoices = order_turnover.withColumn("raw_rating", (F.col("product_orders") / F.col("avg_order")))

    # Flatten Outlier
    ncf_invoices = ncf_invoices.withColumn("rating", F.when(F.col("raw_rating") > 10, 10).otherwise(
                        ncf_invoices.raw_rating))

    # for the purpose of applying normalizer
    ncf_invoices = VectorAssembler(inputCols=["rating"], outputCol="rating_vector").transform(ncf_invoices)

    # minmax normalization methodc
    scaler = MinMaxScaler(min=0.0, max=1.0, inputCol="rating_vector", outputCol="minmax_scaled_rating")
    ncf_invoices = scaler.fit(ncf_invoices).transform(ncf_invoices)
    data = ncf_invoices

    # unlist the vector column
    unlist = F.udf(lambda x: float(list(x)[0]), T.DoubleType())

    w = Window.partitionBy("model_number")

    scaled_result = (F.col("rating") - ((F.min("rating").over(w))-1)) / (F.max("rating").over(w) - ((F.min("rating").over(w))-1))

    ncfmod = data.withColumn("interaction_round", F.round(scaled_result,3))

    ncfmod = ncfmod.withColumn("interaction_round", F.when(F.col('interaction_round') > 0, 1).otherwise(0))
    # unlist the vector column
    unlist = F.udf(lambda x: float(list(x)[0]), T.DoubleType())
    ncfmod = ncfmod.withColumn("interaction", F.round(unlist("minmax_scaled_rating"),6))



    ncfmod = ncfmod.dropna()
    return ncfmod



        
def get_model():
    latent_dim_mf = 32
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    
    num_users = len(ncf_data['customer_id'].unique())
    num_items = len(ncf_data['product_id'].unique())

    # Embedding layers
    mf_user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim_mf, name='mf_user_embedding')
    mf_item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim_mf, name='mf_item_embedding')
    # MF path
    mf_user_latent = Flatten()(mf_user_embedding(user_input))
    mf_item_latent = Flatten()(mf_item_embedding(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])

    # Output layer
    output_layer = Dense(1, activation='sigmoid', kernel_initializer="lecun_uniform", name='output')
    mlp_output = output_layer(mf_vector)

    # Define the model
    model1 = Model(inputs=[user_input, item_input], outputs=mlp_output)

    return model1 
        



def main():
    model  = get_model()
    
    # Create an optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss = tf.keras.losses.BinaryCrossentropy()
                  , metrics=[tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error'),
        tf.keras.metrics.MeanSquaredError(name='mean_squared_error'),
        tf.keras.metrics.RootMeanSquaredError(name='rootMean_squared_error')
    ]
                  ) 
    model._name = "neural_collaborative_filtering"

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1,
                                          restore_best_weights=True)
   # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint = ModelCheckpoint("/visualization/{epoch}mf_model.hdf5",
                                 monitor='loss', verbose=1, save_best_only=False, mode='auto', period=1)
    # Split the data into training and validation sets
    train_user_input, val_user_input, train_item_input, val_item_input, train_ratings, val_ratings = (
        train_test_split(ncf_data['customer_id'], ncf_data['product_id'], ncf_data['interaction'],
                         test_size=0.2, random_state=123))
    # Train the model using the validation set and the EarlyStopping callback
    ncf_mod = model.fit([train_user_input, train_item_input], train_ratings, batch_size=128,
                        epochs=30, verbose=1, validation_data=([val_user_input, val_item_input], val_ratings),
                        callbacks=[es])#, tensorboard_callback, checkpoint])
    
    model.summary()

 
if __name__ == '__main__':
    main()
