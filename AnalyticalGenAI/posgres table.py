import pandas as pd
from sqlalchemy import create_engine

# === CONFIGURATION ===

# Path to your CSV file
csv_file_path = "D:/Visual-FloatChat/smt/sample_gold_layer.csv"

# PostgreSQL connection details
db_user = 'postgres'
db_password = 'sama1234'
db_host = 'localhost'     # or remote host
db_port = '5432'          # default PostgreSQL port
db_name = 'SIH'
table_name = 'sample_gold_layer'  # Target table name in PostgreSQL

# === READ CSV ===

try:
    df = pd.read_csv(csv_file_path)
    print(f"CSV file '{csv_file_path}' read successfully.")
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# === CONNECT TO POSTGRES ===

connection_string = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

try:
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        # === WRITE TO DATABASE ===
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Data written to table '{table_name}' in database '{db_name}'.")
except Exception as e:
    print(f"Error writing to PostgreSQL: {e}")
