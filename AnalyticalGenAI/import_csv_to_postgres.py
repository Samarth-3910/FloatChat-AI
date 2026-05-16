import pandas as pd
from sqlalchemy import create_engine

# Database connection settings
db_user = 'postgres'
db_password = 'sama1234'
db_host = 'localhost'
db_port = '5432'
db_name = 'floatchatAI'

# Create database engine
connection_string = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(connection_string)

# Read CSV file
print("Reading CSV file...")
df = pd.read_csv('sample_gold_layer.csv')
print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
 
# Write to PostgreSQL
print("Writing to PostgreSQL...")
df.to_sql('sample_gold_layer', engine, if_exists='replace', index=False)
print("✅ Successfully created table 'sample_gold_layer' in database 'floatchatAI'!")
print(f"Table has {len(df)} rows and {len(df.columns)} columns")
