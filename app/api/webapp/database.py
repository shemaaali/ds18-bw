# import some libraries
#import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2

import pandas as pd
import numpy as np


df = pd.read_csv("https://raw.githubusercontent.com/bw-airbnb-2/DS/master/airbnb.csv", index_col=0)
print(df.shape)
print(df.head())

DB_NAME = 'airbnb-price'
DB_USER = 'onqmgnvx'
DB_PASSWORD = 'DXqGX7cJBPZnoIs--boAioJLHlbKVGIu'
DB_HOST = 'lallah.db.elephantsql.com'

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER,
                        password=DB_PASSWORD, host = DB_HOST)
                   
print("connection:", conn)
cur = conn.cursor()
print("cursor:", cur)

cur.execute('''CREATE TABLE airbnb_table(Zipcode int,
 Square Feet decimal, 
 Bedrooms decimal,
 Bathrooms decimal,
 Review Scores decimal,
 Rating decimal,
 Accommodates decimal,	
 Cleaning Fee decimal,
 Free Parking decimal,
 Wireless Internet decimal, 
 Cable TV decimal,
 Prop_encoded decimal,
 cancel_encoded decimal,
 Price decimal
  );''')

cur.execute("SELECT * from airbnb_table;")

result_query = cur.fetchall()
print("result_query:",result_query)

#cur.close()
#conn.commit()
#conn.close()

# for loop to insert the whole table
for index, row in df.iterrows():
     cur.execute(f'''INSERT INTO airbnb_table(Zipcode, Square Feet, Bedrooms, Bathrooms, Review Scores, Rating, Accommodates, Cleaning Fee, Free Parking, Wireless Internet, Cable TV, Prop_encoded, cancel_encoded, Price)
VALUES 
( 
  {row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, {row[5]}, {row[6]}, {row[7]}, {row[8]}, {row[9]}, {row[10]}, {row[11]}, {row[11]}
)
    ''')

# create the database with sqlite
SQLALCHEMY_DATABASE_URL = "sqlite:///sql_app.db"
#SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

# make the connection with sqlite
# create a function to return a base 
# Later we will inherit from this class to create each of the database models or classes
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()