import pandas as pd
import time


def load_sample_data(file_path, sample_size=10000):
    """loads a sample subset of json data from the given file path with specified size"""
    try:
        print(f"loading {sample_size} records...")
        start_time = time.time()
        data = pd.read_json(file_path, lines=True, nrows=sample_size)
        print(f"data loaded in {time.time() - start_time:.2f} seconds")
        return data
    except Exception as e:
        print(f"error loading data: {e}")
        return None


def load_full_data(file_path):
    """loads complete json dataset with optimized data types for memory efficiency"""
    try:
        print("loading full dataset...")
        start_time = time.time()
        data = pd.read_json(
            file_path,
            lines=True,
            dtype={"user_id": "int32", "item_id": "int32", "rating": "float32"},
        )
        print(f"full dataset loaded in {time.time() - start_time:.2f} seconds")
        print(f"total records: {len(data):,}")
        print(f"memory usage: {data.memory_usage().sum() / 1024**2:.2f} mb")
        return data
    except Exception as e:
        print(f"error loading data: {e}")
        return None


def load_metadata(file_path):
    """loads json metadata from the specified file path with performance metrics"""
    try:
        print("loading metadata...")
        start_time = time.time()
        metadata = pd.read_json(file_path, lines=True)
        print(f"metadata loaded in {time.time() - start_time:.2f} seconds")
        print(f"total records: {len(metadata):,}")
        print(f"memory usage: {metadata.memory_usage().sum() / 1024**2:.2f} mb")
        return metadata
    except Exception as e:
        print(f"error loading metadata: {e}")
        return None
