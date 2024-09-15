from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

Base = declarative_base()

class UploadMetadata(Base):
    __tablename__ = 'upload_metadata'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    name = Column(String)

    

    data_records = relationship("DataRecord", back_populates="upload_metadata")
    trained_models = relationship("TrainedModel", back_populates="upload_metadata")

class DataRecord(Base):
    __tablename__ = 'data_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity = Column(String)
    year = Column(Integer)
    access_to_electricity = Column(Float)
    access_to_clean_fuels = Column(Float)
    renewable_capacity = Column(Float)
    financial_flows = Column(Float)
    renewable_energy_share = Column(Float)
    electricity_from_fossil_fuels = Column(Float)
    electricity_from_nuclear = Column(Float)
    electricity_from_renewables = Column(Float)
    low_carbon_electricity = Column(Float)
    primary_energy_consumption = Column(Float)
    energy_intensity = Column(Float)
    co2_emissions = Column(Float)
    renewables_percentage = Column(Float)
    gdp_growth = Column(Float)
    gdp_per_capita = Column(Float)
    density = Column(Float)
    land_area = Column(Float)
    latitude = Column(Float)
    longitude = Column(Float)
    
    upload_metadata_id = Column(Integer, ForeignKey('upload_metadata.id'))
    
    upload_metadata = relationship("UploadMetadata", back_populates="data_records")


class TrainedModel(Base):
    __tablename__ = 'trained_models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)  
    mean_squared_error = Column(Float, nullable=True)
    r_squared_score = Column(Float, nullable=True)
    mean_absolute_error = Column(Float, nullable=True)
    training_date = Column(DateTime, default=datetime.utcnow)
    features = Column(String)  
    target = Column(String)  
    upload_metadata_id = Column(Integer, ForeignKey('upload_metadata.id'))
    
    upload_metadata = relationship("UploadMetadata", back_populates="trained_models")


    prediction_history = relationship("PredictionHistory", back_populates="trained_models")


class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    input_data = Column(String)  
    prediction = Column(Float)  
    trained_model_id = Column(Integer, ForeignKey('trained_models.id'))  

    trained_models = relationship("TrainedModel", back_populates="prediction_history")  

# Create the engine and tables
DATABASE_URL = 'postgresql://postgres:12@localhost:5432/TestDB'
engine = create_engine(DATABASE_URL)

Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

print("Tables created successfully.")