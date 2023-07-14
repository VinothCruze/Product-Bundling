from dataset_creation import dataset_creation
from config import dataset_file_path
from bundled import prod_bundle


if __name__ == '__main__':
    inv_ds = dataset_creation()
    inv_ds = inv_ds.repartition(1)
    inv_ds.write.format("parquet").mode('overwrite').save(dataset_file_path)
    results = prod_bundle()



