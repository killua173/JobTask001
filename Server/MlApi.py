import pandas as pd
import asyncio
from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from DbModels import UploadMetadata, DataRecord 


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

async def main():
    upload_id = await insert_data_and_get_upload_id(data)
    print(f"Upload ID: {upload_id}")

if __name__ == "__main__":
    asyncio.run(main())