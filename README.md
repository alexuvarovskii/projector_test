## Projector ML in Production Test


clone the repository


### 1. Train a model from [Kaggle](https://www.kaggle.com/c/commonlitreadabilityprize/overview)

```
    for running this project you need to have python3 installed on your machine
    # install the requirements.txt
    pip isntall -r requirements.txt

    # to run training run 
    python train.py
```

Train script will train a model and save it in the `trainer` folder
Metrics will be counted right after the training process and outputed to terminal window
In the `trainer` folder you will find the model named `pytorch_model.bin`, you can use it for inference

To make predictions on test set run `predict.ipynb` notebook
There upu need to set path to trained model. to the variable `path_to_trained_model`

RMSE on val set in my case is RMSE: 0.5945103809534943

Trained model link [here](https://drive.google.com/file/d/1HOBu8hsvWC-mBho27CAmXOCOHu3_cyXE/view?usp=drive_link)

### 2. Create API with only one endpoint

API is created with FastAPI framework
To run the API you need to run the following command

```
    uvicorn main:app --reload
```

code stored in `main.py` file

But before running it set the path to the trained model in the `main.py` file to the variable `model_path`

it has only one endpoint - `/api/v1/regression` with endpoint `text`
The full path to the endpoint is `http://0.0.0.0:8000/api/v1/regression?text=your_text_here`


### 3. Deploy the API to the cloud
I deployed it to google cloud.
For this you have to create an instance and setup firewall rull to allow traffic on 8000 port.
Then you have to install docker on the instance.

To run container run
```
docker build -t projector-test .
docker run -d -p 8000:8000 projector-test
```

Then you can access the API by the following link

My running endpoint is [here](http://34.66.129.236:8000/api/v1/regression?text=%22hello%20world%22)
