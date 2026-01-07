import os
import pandas as pd
import config
from sqlalchemy import create_engine
import glob

def find_latest_gold_parquet():
    """Finds the 'merged_gold' parquet file in the DataEngineering directory."""
    # Assumptions about path based on file structure scan
    gold_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DataEngineering", "Gold_Data")
    
    # Try to find parquet files
    search_pattern = os.path.join(gold_dir, "*.parquet")
    files = glob.glob(search_pattern)
    
    if not files:
        # Fallback if Gold_Data isn't there, maybe check Silver? 
        # Or maybe the path is relative to notebooks
        print(f"[WARN] No parquet files found in {gold_dir}")
        return None
        
    # Get latest file if multiple
    latest_file = max(files, key=os.path.getmtime)
    print(f"[INFO] Found latest gold data: {latest_file}")
    return latest_file

def load_to_postgres():
    parquet_file = find_latest_gold_parquet()
    if not parquet_file:
        print("[ERROR] Could not find Gold Layer data. Please run the Data Engineering pipeline first.")
        return

    print(f"[INFO] Reading {parquet_file}...")
    try:
        df = pd.read_parquet(parquet_file)
        print(f"[INFO] Data loaded. Shape: {df.shape}")
        
        print("[INFO] Connecting to Database...")
        engine = create_engine(config.DATABASE_URI)
        
        table_name = 'sample_gold_layer'
        print(f"[INFO] Writing to table '{table_name}'...")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print("[SUCCESS] Data pipeline bridge complete. Postgres updated.")
        
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")

if __name__ == "__main__":
    load_to_postgres()
