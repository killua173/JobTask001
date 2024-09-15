from sqlalchemy import create_engine
from  Server.DbModels import Base

DATABASE_URL = 'postgresql://postgres:12@localhost:5432/TestDB'
engine = create_engine(DATABASE_URL)

# Create tables if they don't already exist
Base.metadata.create_all(engine)

print("Tables created successfully.")
