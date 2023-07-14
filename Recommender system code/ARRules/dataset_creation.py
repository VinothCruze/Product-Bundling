"""
Dataset creation
===============

.. module author:: VSSB
.. date:: 2023-07-01

Description
-----------
Dataset creation from existing data sets CASA,CASA MD, PRODUCT_STEP DATASET AND INVOICE

Usage
-----
To use the final dataset for applying the ML models like Frequent Itemset Mining, Neural Collaborative filtering and Alternate Least Square

"""
from config import n_partitions_main, log_file_path
# import libraries & spark intialization
from bdr_tools.wci.spectrumconductor.spark_utils import Spark
app_name = f"BundleRecommender_APP"
spark, sc = Spark().get_spark('app_name')
spark.sparkContext.setLogLevel("ERROR")

from bdr_tools.wdi.wit_data_interface import WitDataInterface
from pyspark.sql import functions as F
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging

# declare variables
s_company_code = "1401"
wdi = WitDataInterface(spark, s_desired_company_code=s_company_code)

spark.conf.set("spark.sql.shuffle.partitions", n_partitions_main)

# Configure logging
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, filemode='w')


# function for creating dataset of active customer,product filtered invoice data
def dataset_creation():
    """
    Function to filter the existing dataframe and creating with the required datacolumns.

            Args:
                --

            Returns:
                Combined Dataframe of CASA, CASA_MD, Invoicem, product step to have active customers,
                active products filtered invoices.
    """
    try:
        # function to have semantic names
        def get_semantic_dataframe(wdi, data_source):
            """Function to get semantic data field name using wdi.

                Args:
                    wdi: wdi interface.
                    data_source: dataframe to be passed to derive the sematic dataframe name.

                Returns:
                    Dataframe with semantic field names.
                """
            sn = wdi.get_semantic_fieldnames(data_source)
            return wdi.get_dataframe(data_source).select([F.col(k).alias(sn[k]) for k in sn])

        # load necessary data sources
        inv_lab = get_semantic_dataframe(wdi, data_source='ds_invoices')
        if not inv_lab:
            logging.error("Problem with the invoice dataframe, check the invoice dataframe import")
            raise SystemExit(1)  # Stop the program execution
        casa_lab = get_semantic_dataframe(wdi, data_source='ds_casa')
        casa_md_lab = get_semantic_dataframe(wdi, data_source='ds_md_customers')
        if not casa_lab:
            logging.error("Problem with the casa_lab dataframe, check the casa_lab dataframe import")
            raise SystemExit(1)  # Stop the program execution
        if not casa_md_lab:
            logging.error("Problem with the casa_md_lab dataframe, check the casa_md_lab dataframe import")
            raise SystemExit(1)  # Stop the program execution
        prod_df = wdi.get_dataframe("ds_md_products_lab")
        if not prod_df:
            logging.error("Problem with the prod_df dataframe, check the prod_df dataframe import")
            raise SystemExit(1)  # Stop the program execution
        # Get last non-empty month from CASA
        last_month = casa_lab.where(casa_lab['active_customer_cr'] == 'X')\
            .agg(F.max(casa_lab['calendar_yearmonth'])).collect()[0][0]

        ## to do: log the file loaded for the timeframe

        if not last_month:
            logging.error("Problem with the last_month variable")
            raise SystemExit(1)  # Stop the program execution
        # filtering the de9 customer data
        # customers = spark.read.format("orc").load(
        #     f'/datalake/WMLA/data/d_1401/data/in/de9data/de9_customer/{last_month}_de9-customer.orc')
        casa_lab_t1 = casa_lab.where((casa_lab['calendar_yearmonth'] == last_month) &
                                     (casa_lab['active_customer_cr'] == 'X')) \
            .drop('customer_sales_view')  # drop this column, because in common with CUST_RAW

        casa_md_lab1 = casa_md_lab.where((casa_md_lab['calendar_yearmonth'] == last_month) &
                                         (casa_md_lab['customer_account_group'] == 'DEBI') &
                                         (casa_md_lab['deletion_flag_for_customer_sales_level'] != 'X') &
                                         (casa_md_lab['customer_status'] == 4)) \
            .drop('potential_lc', 'sales_organization','potential_lc', 'sales_organization','purity_degree_sales_sector','company_code',' customer_circle','statistics_currency',
         'source_system_id',' purity_degree_sales_channel','purity_degree_segmentation')  # drop these columns, because in common with CASA_RAW
        # Check if the count of records in casa_md_lab1 is less than the count of records in casa_md_lab
        if casa_md_lab1.count() >= casa_md_lab.count():
            logging.error("The count of records in casa_md_lab1 is not less than the count of records in casa_md_lab, check the filtering")
            raise SystemExit(1)  # Stop the program execution
        # joining the CASA lab and master data
        casa = casa_lab_t1.join(casa_md_lab1, [casa_lab_t1.customer == casa_md_lab1.customer_sales_view])
        casa1 = casa.select('customer', 'sales_region', 'customer_account_group', 'customer_status',
                            'active_customer_cr', 'sml_turnover_class_cr', 'sales_channel')

        casa_current = casa1.filter(
            (F.col("customer_account_group") == "DEBI")  # customers with DEBIT status(reason for picking)
            & (F.col("active_customer_cr") == "X")  # active customers
            & (F.col("customer_status") == 4)  # not marked as deleted
            & ((F.col('sales_region') >= "000000000011") &  # region filter from 11-18
               (F.col("sales_region") <= "000000000018"))
            & (F.trim(F.col("customer")) != '')  # for filtering the values that are empty
            & (F.col("sales_channel") == "RV"))  # filtering the sales channel RV
        logging.info('customer dataset is filtered with active customers and neccessary filters successfully')

        # # customers from de9
        # customers_r = customers.select("KNA1_KUNNR", "KNB5_ZZACTMS")
        # casa_fil = casa_current.join(customers_r, on=casa_current.customer == customers_r.KNA1_KUNNR, how="left")
        # casa_fil = casa_fil.filter(F.col("KNB5_ZZACTMS") <= 2)  # filter reliable customers

        # filter for active customers and their active region
        casa_curr_fil = casa_current.select(F.col('customer'), F.col('sales_region'))

        # region naming
        casa_curr_fil = casa_curr_fil.withColumn('region',
                                                 F.when(casa_curr_fil.sales_region == '000000000018',
                                                        F.lit('Region-18'))
                                                 .when(casa_curr_fil.sales_region == '000000000017', F.lit('Region-17'))
                                                 .when(casa_curr_fil.sales_region == '000000000016', F.lit('Region-16'))
                                                 .when(casa_curr_fil.sales_region == '000000000015', F.lit('Region-15'))
                                                 .when(casa_curr_fil.sales_region == '000000000014', F.lit('Region-14'))
                                                 .when(casa_curr_fil.sales_region == '000000000013', F.lit('Region-13'))
                                                 .when(casa_curr_fil.sales_region == '000000000012', F.lit('Region-12'))
                                                 .when(casa_curr_fil.sales_region == '000000000011', F.lit('Region-11'))
                                                 .otherwise(casa_curr_fil.sales_region))

        # drop duplicates
        casa_curr_fil.dropDuplicates()

        # filter invoice dataset with active customers
        columns = ('customer', 'sales_region')
        inv_lab = inv_lab.drop(*columns)
        inv_lab = inv_lab.join(casa_curr_fil, [inv_lab.soldtoparty == casa_curr_fil.customer], 'inner')

        logging.info('invoice dataset merged with active customers successfully')

        # filter to remove inactive products history
        prod_active = prod_df.filter((prod_df.n_ac_status == 5) & (prod_df.b_ac_is_in_eshop == 'true')
                              & (F.col('partition_value') == "ACTUAL"))

        # columns to drop from product dataset
        cols = ("s_mc_product_model_name", "s_mc_name", "s_mc_label", "s_mc_description", "s_mc_product_model_label",
                "s_mc_ShortTextUsp", "s_mc_keywords", "s_p_number", "s_p_CompetitorProduct", "s_pc_label"
                , "s_a_article_number", "s_pc_product_label", "s_a_article_number")

        prod_active = prod_active.drop(*cols)
        prod_active = prod_active.dropDuplicates()

        # renaming the products column names to understandable
        prod_active = prod_active.withColumnRenamed("n_product_area_nr", "product_field") \
            .withColumnRenamed("n_product_group_nr", "product_group") \
            .withColumnRenamed("n_product_class_nr", "product_class") \
            .withColumnRenamed("n_product_family_nr", "product_family") \
            .withColumnRenamed("n_product_model_nr", "productmodel") \
            .withColumnRenamed("s_p_product_number", "article_number")
        # filter invoice dataset with active products
        invoice_active = prod_active.join(inv_lab,[(prod_active.article_number == inv_lab.product)]
                                                    ,'inner')
        logging.info('invoice dataset merged with active products successfully')
        col_names = ['calendar_day', 'soldtoparty', 'order_date', 'order_reason_statistic', 'order_number',
                     'sales_sector_of_customer',
                     'business_code', 'contact_point', 'customer_hierarchy_level_1', 'customer_hierarchy_level_2',
                     'customer_hierarchy_level_3', 'custturnover_inv',
                     'turnover_inv', 'invoice_quantity_in_sales_unit_inv', 'product_field_int_product',
                     'product_group_int_product', 'product_class_int_product',
                     'product_family_int_product', 'productmodel', 'product', 'sales_region']

        invoice_filt = invoice_active.select(*col_names)

        # filter invoices that are na for the model number
        invoice_filt = invoice_filt.na.drop(subset=["productmodel"])

        new_names = ["calendar_day", "customernumber", "order_date", "order_reason", "sales_receipt",
                     "customerdivision",
                     'customerindustry', 'custcontact', 'cchierl1', 'cchierl2', 'cchierl3', 'custturnover',
                     'sales_value', 'soldnumber', "product_field", "product_group", 'product_class', "product_family",
                     "model_number", "product_number", 'region']

        invoice_rn = invoice_filt.toDF(*new_names)
        # filter out null invoice values
        invoice_rn = invoice_rn.filter(F.col("sales_receipt").isNotNull())
        # invoice_rn.show(n=5,vertical=True, truncate=False)
        invoice_rn = invoice_rn.withColumn('sales_year', F.year(invoice_rn.calendar_day))
        invoice_rn = invoice_rn.withColumn('sales_month', F.month(invoice_rn.calendar_day))
        invoice_rn = invoice_rn.withColumn('customer_area',
                                           F.when(invoice_rn.customerdivision == 'AU', F.lit('Auto'))
                                           .when(invoice_rn.customerdivision == 'WO', F.lit('Wood'))
                                           .when(invoice_rn.customerdivision == 'ME', F.lit('Metal'))
                                           .when(invoice_rn.customerdivision == 'CO', F.lit('Construction'))
                                           .when(invoice_rn.customerdivision == 'CA', F.lit('Cargo'))
                                           .when(invoice_rn.customerdivision == 'CP', F.lit('ConstrSite project'))
                                           .when(invoice_rn.customerdivision == 'EL', F.lit('House Engineering'))
                                           .when(invoice_rn.customerdivision == 'MA',
                                                 F.lit('Engineering workshop (Betriebswerkstatt)'))
                                           .otherwise(invoice_rn.customerdivision))
        invoice_rn = invoice_rn.na.drop(subset=["customer_area"]).dropDuplicates()
        logging.info('invoice dataset created successfully')
        # filter negative values
        invoice_na = invoice_rn.filter(~F.col('sales_value').like('%-%'))
        invoice_na = invoice_na.dropDuplicates()
        return invoice_na
    except Exception as e:
        logging.error('Error occurred in dataset creation: {}'.format(e))
        return e



