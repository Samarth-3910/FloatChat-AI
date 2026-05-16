import pandas as pd
from sqlalchemy import create_engine

import config

# Create database engine
engine = create_engine(config.DATABASE_URI)

# Read CSV file
print("Reading CSV file...")
df = pd.read_csv('sample_gold_layer.csv')
print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
 
# Write to PostgreSQL
print("Writing to PostgreSQL...")
df.to_sql('sample_gold_layer', engine, if_exists='replace', index=False)
print("✅ Successfully created table 'sample_gold_layer' in database 'floatchatAI'!")
print(f"Table has {len(df)} rows and {len(df.columns)} columns")
