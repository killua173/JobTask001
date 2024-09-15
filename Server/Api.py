import pandas as pd
import os
import pickle
from typing import Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
import io
from fastapi.responses import StreamingResponse
from DbModels import UploadMetadata, DataRecord, TrainedModel, PredictionHistory
from sqlalchemy.orm import selectinload
from sqlalchemy import select, desc
from fastapi.encoders import jsonable_encoder

DATABASE_URL = 'postgresql+asyncpg://postgres:12@localhost:5432/TestDB'

app = FastAPI()

# Database session dependency
async def get_db():
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
    await engine.dispose()

class PredictionInput(BaseModel):
    entity: str
    year: int
    energy_intensity: float
    electricity_from_renewables: float
    latitude: float
    primary_energy_consumption: float
    longitude: float
    access_to_clean_fuels: float
    low_carbon_electricity: float
    access_to_electricity: float
    gdp_per_capita: float

async def upload(df: pd.DataFrame) -> Optional[int]:
    df[r'Density\n(P/Km2)'] = pd.to_numeric(df[r'Density\n(P/Km2)'], errors='coerce')

    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        async with session.begin():
            try:
                new_upload = UploadMetadata(upload_date=datetime.utcnow())
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
    missing_percentage = data.isnull().sum() / len(data) * 100
    columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index
    data = data.drop(columns=columns_to_drop)

    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if abs(data[column].skew()) > 0.5:
                data[column].fillna(data[column].median(), inplace=True)
            else:
                data[column].fillna(data[column].mean(), inplace=True)

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

        cleaned_df = await clean_data(df)  
        return cleaned_df

    await engine.dispose()

async def train_and_evaluate_model(df: pd.DataFrame, upload_metadata_id: int, target: str = 'renewable_capacity', model_name: str='Model') -> None:
    X = df.drop(columns=[target, 'id'])
    y = df[target]
    lsfeatures = X.columns.tolist()
    stfeatures = json.dumps(lsfeatures)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'MLmodels')

    if 'entity' in stfeatures:
        label_encoder = LabelEncoder()
        X['Entity_Encoded'] = label_encoder.fit_transform(X['entity'])
        X = X.drop(columns=['entity'])
        encoder_path = os.path.join(model_dir, f'{model_name}_{upload_metadata_id}_encoder.pkl')
        with open(encoder_path, 'wb') as encoder_file:
             pickle.dump(label_encoder, encoder_file)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\nFinal Model Performance on Test Set:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    rf_model.fit(X_scaled, y)




    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    scaler_path = os.path.join(model_dir, f'{model_name}_{upload_metadata_id}_scaler.pkl')
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    model_path = os.path.join(model_dir, f'{model_name}_{upload_metadata_id}.pkl')
    print(f"Saving model to: {model_path}")  
    with open(model_path, 'wb') as model_file:
        pickle.dump(rf_model, model_file)

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



async def check_existing_model(upload_id: int) -> bool:
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        stmt = select(TrainedModel).where(TrainedModel.upload_metadata_id == upload_id)
        result = await session.execute(stmt)
        existing_model = result.scalar_one_or_none()

    await engine.dispose()
    return existing_model is not None


async def predict(session: AsyncSession, upload_metadata_id: int, input_data: dict):
    stmt = select(TrainedModel).where(TrainedModel.upload_metadata_id == upload_metadata_id)
    result = await session.execute(stmt)
    trained_model = result.scalar_one_or_none()

    if not trained_model:
        raise ValueError(f"No trained model found for upload_metadata_id: {upload_metadata_id}")

    model_name = trained_model.model_name

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'MLmodels', f'{model_name}.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    features = json.loads(trained_model.features)
    input_df = pd.DataFrame([input_data])

    for feature in features:
        if feature not in input_df.columns:
            raise ValueError(f"Missing feature in input data: {feature}")

    input_df = input_df[features]


    if 'entity' in features:
        encoder_path = os.path.join(script_dir, 'MLmodels', f'{model_name}_encoder.pkl')
        with open(encoder_path, 'rb') as encoder_file:
             label_encoder = pickle.load(encoder_file)
        input_df['Entity_Encoded'] = label_encoder.transform(input_df['entity'])
        input_df = input_df.drop(columns=['entity'])

    scaler_path = os.path.join(script_dir, 'MLmodels', f'{model_name}_scaler.pkl')
    with open(scaler_path, 'rb') as scaler_file:
     scaler = pickle.load(scaler_file)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    new_prediction = PredictionHistory(
        prediction_date=datetime.utcnow(),
        input_data=json.dumps(input_data),
        prediction=float(prediction),
        trained_model_id=trained_model.id
    )
    session.add(new_prediction)
    await session.commit()

    return prediction

@app.post("/upload-data")
async def upload_data_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    upload_id = await upload(df)
    if upload_id is None:
        raise HTTPException(status_code=500, detail="Failed to upload data")
    return {"message": "Data uploaded successfully", "upload_id": upload_id}

@app.post("/train-model")
async def train_model_endpoint(upload_id: int, model_name: str = "Model", target: str = "renewable_capacity"):
    # Check if a model already exists for this upload_id
    model_exists = await check_existing_model(upload_id)
    if model_exists:
        return {"message": "Model already trained for this upload_id"}

    df = await process_data(upload_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the given upload_id")
    
    await train_and_evaluate_model(df, upload_id, target, model_name)
    return {"message": "Model trained successfully"}

@app.post("/predict")
async def predict_endpoint(upload_id: int, input_data: PredictionInput, db: AsyncSession = Depends(get_db)):
    try:
        prediction = await predict(db, upload_id, input_data.dict())
        return {"prediction": prediction}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/get-data")
async def get_data_endpoint(country: str, db: AsyncSession = Depends(get_db)):
    stmt = select(DataRecord).where(DataRecord.entity == country)
    result = await db.execute(stmt)
    records = result.scalars().all()
    
    if not records:
        raise HTTPException(status_code=404, detail=f"No data found for country: {country}")
    
    return jsonable_encoder(records)
@app.get("/models")
async def get_models(db: AsyncSession = Depends(get_db)):
    stmt = select(TrainedModel)
    result = await db.execute(stmt)
    models = result.scalars().all()

    if not models:
        raise HTTPException(status_code=404, detail="No models found in the database")

    return jsonable_encoder(models)

@app.get("/predictions")
async def get_predictions(db: AsyncSession = Depends(get_db)):
    stmt = select(PredictionHistory).order_by(desc(PredictionHistory.prediction_date)).limit(25)
    result = await db.execute(stmt)
    predictions = result.scalars().all()

    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions found in the database")

    return jsonable_encoder(predictions)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)