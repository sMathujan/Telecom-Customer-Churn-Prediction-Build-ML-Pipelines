
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

from src.data_cleaning import TelcoDataCleaner
from src.data_ingestion import DataIngestorCSV
from src.handle_missing_values import DropMissingValuesStrategy
from src.outlier_detection import IQROutlierDetection, OutlierDetector
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

    # 2. Clean (TotalCharges → numeric, normalise service categories, drop ID)
    clean_cfg = get_cleaning_config()
    cleaner = TelcoDataCleaner(clean_cfg["coerce_numeric"], clean_cfg["category_normalisation"],
                                   clean_cfg["normalise_columns"], cols["drop_columns"])
    df = cleaner.clean(df)
    print("Cleaned DF")
    print(df.head())

    # 3. Missing values (the 11 tenure-0 rows with blank TotalCharges)
    before = len(df)
    mv_cfg = get_missing_values_config()
    df = DropMissingValuesStrategy(mv_cfg["critical_columns"]).handle(df)
    print("Missing Values Handled DF")
    print({"rows_removed": before - len(df)})
    # log_stage_metrics(df, "missing_handled", {"rows_removed": before - len(df)})

    # 4. Outliers (report-only by default)
    out_cfg = get_outlier_config()
    n_outliers = 0
    if out_cfg.get("enabled", True):
        df, n_outliers = OutlierDetector(IQROutlierDetection(out_cfg.get("iqr_multiplier", 1.5))) \
            .handle_outliers(df, out_cfg["columns"], out_cfg.get("handling_method", "report"))
    print("Outlier Detection")
    print({"outlier_rows": n_outliers})
    # log_stage_metrics(df, "outliers_checked", {"outlier_rows": n_outliers})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the churn data pipeline")
    parser.add_argument("--force", action="store_true", help="Rebuild all processed artifacts")
    args = parser.parse_args()

    result = data_pipeline(force_rebuild=args.force)
    print("\n" + "=" * 60)
    print("📊 DATA PIPELINE SUMMARY")
