# Energy Prediction Project

This project aims to predict renewable energy capacity using machine learning techniques based on various global sustainable energy indicators.
# Getting Started

To get started with the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. Install all dependencies:
   ```bash
   pip install sqlalchemy pandas numpy scikit-learn fastapi pydantic uvicorn matplotlib seaborn
   ```

3. Ensure you are using Python 3.8.0.

4. Update the database connection string in both `create_tables.py` and `api.py`:
   - Open both `create_tables.py` and `api.py` files.
   - In each file, locate the `DATABASE_URL` variable.
   - Change it to match your database setup. For example:
     ```python
     DATABASE_URL = 'postgresql+asyncpg://your_username:your_password@your_host:your_port/your_database'
     ```
   - Make sure to replace `your_username`, `your_password`, `your_host`, `your_port`, and `your_database` with your actual database credentials and details.
   - Ensure that the `DATABASE_URL` is identical in both files.

5. After the dependencies are installed and the database strings are updated, create the necessary database tables by running:
   ```bash
   python create_tables.py
   ```

6. Navigate to the `servers` directory and start the API server:
   ```bash
   python api.py
   ```

7. The API endpoints should now be available and ready to use.

Note: `MachineLearningTest` has been used to test machine learning models before implementing them into the API.



## API Endpoints

The FastAPI application provides the following endpoints:

1. `/upload-data`: POST endpoint to upload a CSV file of the dataset.
   - Accepts a CSV file upload.
   - Returns an upload ID for the uploaded data.

2. `/train-model`: POST endpoint to train a machine learning model on the uploaded data.
   - Parameters:
     - `upload_id`: Integer ID of the uploaded dataset to use for training.
     - `model_name`: (Optional) Name for the model. Default is "Model".
     - `target`: (Optional) Target variable for prediction. Default is "renewable_capacity".
   - Trains a new model if one doesn't exist for the given upload_id.

3. `/predict`: POST endpoint to make predictions using a trained model.
   - Parameters:
     - `upload_id`: Integer ID of the upload associated with the trained model.
     - `input_data`: JSON object containing the input features for prediction.
   - Returns the prediction result.

4. `/get-data`: GET endpoint to retrieve energy data for a specific country.
   - Parameters:
     - `country`: Name of the country to retrieve data for.
   - Returns a CSV file with the country's data.

5. `/models`: GET endpoint to retrieve information about all trained models.
   - Returns a list of all trained models in the database.

6. `/predictions`: GET endpoint to retrieve recent prediction history.
   - Returns the 25 most recent predictions made by the models.






## Database Schema

The project uses SQLAlchemy ORM with the following main tables:

1. **UploadMetadata**: Stores information about data uploads
2. **DataRecord**: Stores individual data points from the dataset
3. **TrainedModel**: Stores information about trained models
4. **PredictionHistory**: Stores history of predictions made
   
## Dataset Description

The dataset used in this project is the "Global Data on Sustainable Energy" which includes information on various energy-related indicators for countries worldwide. Key features include:

- Access to electricity
- Access to clean fuels for cooking
- Renewable electricity generating capacity per capita
- Financial flows to developing countries
- Renewable energy share in total final energy consumption
- Electricity production from various sources (fossil fuels, nuclear, renewables)
- Low-carbon electricity percentage
- Primary energy consumption per capita
- Energy intensity level
- CO2 emissions
- GDP growth and GDP per capita
- Population density and land area
- Geographical coordinates (latitude and longitude)

This comprehensive dataset allows for in-depth analysis and prediction of renewable energy capacity based on multiple socio-economic and geographical factors.

