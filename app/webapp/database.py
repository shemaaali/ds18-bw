# import some libraries
#import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2

from api import predict, viz

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