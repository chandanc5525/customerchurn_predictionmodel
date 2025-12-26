import pandas as pd
import yaml
from logger import get_logger

logger = get_logger(__name__)

def load_data():
    cfg = yaml.safe_load(open("src/config/config.yaml"))
    df = pd.read_csv(cfg["data_path"])
    logger.info("Data Ingestion Done Successfully")
    return df, cfg
