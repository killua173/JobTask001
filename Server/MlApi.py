import pandas as pd
import asyncio
from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from DbModels import UploadMetadata, DataRecord 
from sqlalchemy.future import select
import numpy as np


DATABASE_URL = 'postgresql+asyncpg://postgres:12@localhost:5432/TestDB'

file_path = r'C:\Users\hamza\Downloads\archive (1)\global-data-on-sustainable-energy (1).csv'
data = pd.read_csv(file_path)

async def insert_data_and_get_upload_id(df) -> Optional[int]:
    df[r'Density\n(P/Km2)'] = pd.to_numeric(df[r'Density\n(P/Km2)'], errors='coerce')

    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        async with session.begin():
            try:
                # 1. Insert into upload_metadata
                new_upload = UploadMetadata(
                    upload_date=datetime.utcnow()
                )
                session.add(new_upload)
                await session.flush()  

            
                upload_id = new_upload.id

                
                for _, row in df.iterrows():
                    new_record = DataRecord(
                        entity=row['Entity'],
                        year=row['Year'],
                        access_to_electricity=row[r'Access to electricity (% of population)'],
                        access_to_clean_fuels=row[r'Access to clean fuels for cooking'],
                        renewable_capacity=row[r'Renewable-electricity-generating-capacity-per-capita'],
                        financial_flows=row[r'Financial flows to developing countries (US $)'],
                        renewable_energy_share=row[r'Renewable energy share in the total final energy consumption (%)'],
                        electricity_from_fossil_fuels=row[r'Electricity from fossil fuels (TWh)'],
                        electricity_from_nuclear=row[r'Electricity from nuclear (TWh)'],
                        electricity_from_renewables=row[r'Electricity from renewables (TWh)'],
                        low_carbon_electricity=row[r'Low-carbon electricity (% electricity)'],
                        primary_energy_consumption=row[r'Primary energy consumption per capita (kWh/person)'],
                        energy_intensity=row[r'Energy intensity level of primary energy (MJ/$2017 PPP GDP)'],
                        co2_emissions=row[r'Value_co2_emissions_kt_by_country'],
                        renewables_percentage=row[r'Renewables (% equivalent primary energy)'],
                        gdp_growth=row[r'gdp_growth'],
                        gdp_per_capita=row[r'gdp_per_capita'],
                        density=row[r'Density\n(P/Km2)'],
                        land_area=row[r'Land Area(Km2)'],
                        latitude=row[r'Latitude'],
                        longitude=row[r'Longitude'],
                        upload_metadata_id=upload_id
                    )
                    session.add(new_record)

                await session.commit()
                print(f"Data inserted successfully! Upload ID: {upload_id}")
                return upload_id

            except Exception as e:
                
                print(f"An error occurred: {str(e)}")
                return None

    await engine.dispose()






async def clean_data(data: pd.DataFrame, missing_threshold: float = 35, outlier_threshold: float = 3) -> pd.DataFrame:

    # Print initial missing percentage and data types
    missing_percentage = data.isnull().sum() / len(data) * 100
    print("Missing Percentage Before:")
    print(missing_percentage)


    # Drop columns with more than the specified percentage of missing values
    columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index
    data = data.drop(columns=columns_to_drop)
    print(f"\nColumns dropped due to more than {missing_threshold}% missing values:")
    print(columns_to_drop.tolist())


    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if abs(data[column].skew()) > 0.5:
                data[column].fillna(data[column].median(), inplace=True)
            else:
                data[column].fillna(data[column].mean(), inplace=True)

 
    """ def remove_outliers(df):
        columns = df.select_dtypes(include=[np.number]).columns
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            print(f"Column: {col}")
            print(f"Lower Bound: {lower_bound}")
            print(f"Upper Bound: {upper_bound}\n")
            
        return df

    data_clean = remove_outliers(data) """

    print("\nNull Values Sum:")
    print(data.isnull().sum())


    return data

async def check_id(session: AsyncSession, id: int) -> bool:
    stmt = select(UploadMetadata).where(UploadMetadata.id == id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None

async def retrieve_data(session: AsyncSession, id: int) -> pd.DataFrame:
    stmt = select(DataRecord).where(DataRecord.upload_metadata_id == id)
    result = await session.execute(stmt)
    records = result.scalars().all()

    if not records:
        print(f"No DataRecords found for upload_metadata_id: {id}")
        return pd.DataFrame()

    # Convert to DataFrame for easier handling
    df = pd.DataFrame([
        {
            'id': r.id,
            'entity': r.entity,
            'energy_intensity': r.energy_intensity,
            'electricity_from_renewables': r.electricity_from_renewables,
            'latitude': r.latitude,
            'primary_energy_consumption': r.primary_energy_consumption,
            'longitude': r.longitude,
            'access_to_clean_fuels': r.access_to_clean_fuels,
            'low_carbon_electricity': r.low_carbon_electricity,
            'access_to_electricity': r.access_to_electricity,
            'gdp_per_capita': r.gdp_per_capita,
            'year': r.year,
            'renewable_capacity': r.renewable_capacity
        } for r in records
    ])

    return df


async def process_data(id: int) -> pd.DataFrame:
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        if not await check_id(session, id):
            print(f"No UploadMetadata found for id: {id}")
            return pd.DataFrame() 

        df = await retrieve_data(session, id)

        if df.empty:
            return pd.DataFrame()  

        # Clean the data
        cleaned_df = await clean_data(df)  
        return cleaned_df
    

    await engine.dispose()


async def main():
  
    df = await process_data(1)
    print(f"DataFrame: {df}")
    

""" upload_id = await insert_data_and_get_upload_id(data)
  print(f"Upload ID: {upload_id}") """


if __name__ == "__main__":
    asyncio.run(main())