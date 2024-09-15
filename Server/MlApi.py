import pandas as pd
import os
import pickle
import asyncio
from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from DbModels import UploadMetadata, DataRecord ,TrainedModel,PredictionHistory
from sqlalchemy.future import select
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
from sklearn.preprocessing import LabelEncoder

DATABASE_URL = 'postgresql+asyncpg://postgres:12@localhost:5432/TestDB'

file_path = r'C:\Users\hamza\Downloads\archive (1)\global-data-on-sustainable-energy (1).csv'
data = pd.read_csv(file_path)

async def upload(df) -> Optional[int]:
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


async def train_and_evaluate_model(df: pd.DataFrame, upload_metadata_id: int, target: str = 'renewable_capacity',model_name: str='Model') -> None:
    X = df.drop(columns=[target, 'id'])
    y = df[target]
    lsfeatures= X.columns.tolist()
    stfeatures =json.dumps(lsfeatures)

    # Convert categorical variables to dummy variables
    if 'entity' in stfeatures:
      label_encoder = LabelEncoder()
      X['Entity_Encoded'] = label_encoder.fit_transform(X['entity'])
      X = X.drop(columns=['entity'])

    # Scale the feature variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Fit the model
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\nFinal Model Performance on Test Set:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")


    rf_model.fit(X_scaled, y)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'MLmodels')



    # Ensure MLmodels directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model
    model_path = os.path.join(model_dir, f'{model_name}_{upload_metadata_id}.pkl')
    print(f"Saving model to: {model_path}")  
    with open(model_path, 'wb') as model_file:
        pickle.dump(rf_model, model_file)


    # Save the metrics to the database
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        async with session.begin():
            evaluation = TrainedModel(
                upload_metadata_id=upload_metadata_id,
                model_name=f'{model_name}_{upload_metadata_id}',
                mean_squared_error=mse,
                r_squared_score=r2,
                mean_absolute_error=mae,
                training_date=datetime.utcnow(),
                features=stfeatures,
                target=target,
            )
            session.add(evaluation)
            await session.commit()

    await engine.dispose()


async def train_model(id: int, model_name: str = 'Model', target: str = 'renewable_capacity') -> None:
    # Initialize the database connection and session
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
       
       df = await process_data(1)
       await train_and_evaluate_model(df, id, target, model_name)
    
    # Dispose the engine after use
    await engine.dispose()









async def predict(session: AsyncSession, upload_metadata_id: int, input_data: dict):
    # Step 1: Get the model name by searching upload_metadata_id
    stmt = select(TrainedModel).where(TrainedModel.upload_metadata_id == upload_metadata_id)
    result = await session.execute(stmt)
    trained_model = result.scalar_one_or_none()

    if not trained_model:
        raise ValueError(f"No trained model found for upload_metadata_id: {upload_metadata_id}")

    model_name = trained_model.model_name

    # Step 2: Load the model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'MLmodels', f'{model_name}.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Step 3: Prepare input data for prediction
    features = json.loads(trained_model.features)
    input_df = pd.DataFrame([input_data])

    # Ensure all features are present in input_data
    for feature in features:
        if feature not in input_df.columns:
            raise ValueError(f"Missing feature in input data: {feature}")




    # Reorder columns to match the training data
    input_df = input_df[features]
    if 'entity' in features:
          label_encoder = LabelEncoder()
          input_df['Entity_Encoded'] = label_encoder.fit_transform(input_df['entity'])
          input_df = input_df.drop(columns=['entity'])

    # Scale the input data
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)

    # Step 4: Make prediction
    prediction = model.predict(input_scaled)[0]

    # Step 5: Save the prediction to the database
    new_prediction = PredictionHistory(
        prediction_date=datetime.utcnow(),
        input_data=json.dumps(input_data),
        prediction=float(prediction),
        trained_model_id=trained_model.id
    )
    session.add(new_prediction)
    await session.commit()

    return prediction


















async def main():
#  await train_model(1)
    
 

 
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Test input data
        input_data = {
            'entity': 'United States',
            'year': 2022,
            'energy_intensity': 5.2,
            'electricity_from_renewables': 834.0,
            'latitude': 37.0902,
            'primary_energy_consumption': 75000,
            'longitude': -95.7129,
            'access_to_clean_fuels': 100.0,
            'low_carbon_electricity': 40.0,
            'access_to_electricity': 100.0,
            'gdp_per_capita': 63000
        }

        try:
            # Assuming the upload_metadata_id is 1. Replace with the correct id if different.
            prediction = await predict(session, upload_metadata_id=1, input_data=input_data)
            print(f"Prediction result: {prediction}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())

print("Current Working Directory:", os.getcwd())
