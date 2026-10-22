
import argparse
import json
import os
import sys
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.data_ingestion import DataIngestorCSV
from utils.config import (get_cleaning_config, get_columns, get_encoding_config,
                          get_missing_values_config, get_outlier_config, get_path,
                          get_splitting_config)
from utils.logger import banner, get_logger
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags

logger = get_logger(__name__)


def data_pipeline(force_rebuild: bool = False) -> Dict[str, pd.DataFrame]:

    raw_path = get_path("raw_data")
    print("Raw Path: ", raw_path)
    cols = get_columns()
    print("Columns: ", cols)
    target = cols["target"]
    print("Target Column: ", target)


    df = DataIngestorCSV(required_columns=cols["raw_feature_columns"] + [target]).ingest(raw_path)
    print("DF")
    print(df.head())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the churn data pipeline")
    parser.add_argument("--force", action="store_true", help="Rebuild all processed artifacts")
    args = parser.parse_args()

    result = data_pipeline(force_rebuild=args.force)
    print("\n" + "=" * 60)
    print("📊 DATA PIPELINE SUMMARY")
