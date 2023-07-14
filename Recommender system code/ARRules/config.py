# config.py

# date time calculation
from datetime import datetime, timedelta
# Spark configuration
app_name = "FQM_APP"
n_partitions = 2
n_partitions_main = 20

# Dataset file path
dataset_file_path = "/datalake/WMLA/data/d_1401/projects/d_00209769/code/dev/src/Bundle/parquet/invoice_finalpc.parquet"
log_file_path = "/datalake/WMLA/data/d_1401/projects/d_bdo/data/logs/bundle_' + time.strftime('%%Y%%m%%d%%H%%M%%S') + '.log'"
file_path = "/datalake/WMLA/data/d_1401/projects"
# Date range
current_date = datetime.now().date()
first_date_of_month = current_date.replace(day=1)
date_12_months_back = first_date_of_month.replace(year=first_date_of_month.year - 1)

current_date_str = first_date_of_month.strftime('%Y-%m-%d')
past_date_str = date_12_months_back.strftime('%Y-%m-%d')

#FQM Parameters
minSupport = 0.001
minConfidence = 0.50
lift = 1

# Customer areas
customer_areas = ['Auto', 'Construction', 'Cargo', 'House Engineering', 'Metal', 'Wood',
'ConstrSite project', 'Engineering workshop (Betriebswerkstatt)']  # Add more customer areas if needed
