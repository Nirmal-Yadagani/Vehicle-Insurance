from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, InsuranceSubscriptionClassifier
from src.pipline.training_pipeline import TrainPipeline


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')


origins = ['*']

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*'])


class DataForm:
    """
    Handles the parsing and extraction of data from the HTML form request.
    
    This class acts as a data transfer object (DTO) between the web form 
    and the internal Prediction Pipeline.
    """
    def __init__(self, request: Request):
        """
        Initializes the DataForm with the FastAPI Request object.
        """
        self.request: Request = request
        self.Gender: Optional[str] = None
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[int] = None
        self.Previously_Insured: Optional[int] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[float] = None
        self.Vehicle_Age: Optional[str] = None
        self.Vehicle_Damage: Optional[str] = None


    async def get_vehicle_data(self):
        """
        Parses the form data from the asynchronous request and populates class attributes.
        """
        form = await self.request.form()
        self.Gender = form.get("Gender")
        self.Age = int(form.get("Age"))
        self.Driving_License = int(form.get("Driving_License"))
        self.Region_Code = int(form.get("Region_Code"))
        self.Previously_Insured = int(form.get("Previously_Insured"))
        self.Annual_Premium = float(form.get("Annual_Premium"))
        self.Policy_Sales_Channel = float(form.get("Policy_Sales_Channel"))
        self.Vintage = float(form.get("Vintage"))
        self.Vehicle_Age = form.get("Vehicle_Age")
        self.Vehicle_Damage = form.get("Vehicle_Damage")



@app.get('/', tags=['authentication'])
async def index(request: Request):
    """
    Renders the main index page (vehicledata.html).

    Args:
        request (Request): The incoming FastAPI request.

    Returns:
        TemplateResponse: The rendered HTML page.
    """
    return templates.TemplateResponse('vehicledata.html', {'request': request, 'context': "Rendering"})


@app.get('/train')
async def trainRouteClient():
    """
    API endpoint to trigger the end-to-end Training Pipeline.

    This route initializes the TrainPipeline and executes all stages 
    from Data Ingestion to Model Pushing.

    Returns:
        Response: Success or failure message of the training process.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training Successful!!")
    
    except Exception as e:
        return Response(f"Error Occurred! {e}")
    

@app.post('/')
async def predictRouteClient(request: Request):
    """
    API endpoint to handle model inference.

    Takes form data, converts it into a DataFrame, executes the prediction 
    logic, and returns the result to the HTML template.

    Args:
        request (Request): The POST request containing vehicle/user data.

    Returns:
        TemplateResponse: The HTML page updated with the prediction status.
    """
    try:
        form = DataForm(request=request)
        await form.get_vehicle_data()

        vehicle_data = VehicleData(Gender= form.Gender,
                                Age = form.Age,
                                Driving_License = form.Driving_License,
                                Region_Code = form.Region_Code,
                                Previously_Insured = form.Previously_Insured,
                                Annual_Premium = form.Annual_Premium,
                                Policy_Sales_Channel = form.Policy_Sales_Channel,
                                Vintage = form.Vintage,
                                Vehicle_Age = form.Vehicle_Age,
                                Vehicle_Damage = form.Vehicle_Damage
                                )
        
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        model_predictor = InsuranceSubscriptionClassifier()

        status = f'Response-{model_predictor.predict(vehicle_df)}'

        return templates.TemplateResponse('vehicledata.html', {'request': request, 'context': status})
    
    except Exception as e:
        return {'status':False, 'error':f'{e}'}

if __name__=="__main__":
    app_run(app=app, host=APP_HOST, port=APP_PORT)
        
