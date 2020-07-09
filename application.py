from flask import Flask, request, Markup, make_response
from flask_restful import Resource, Api
import urllib
from urllib.parse import quote
import pandas as pd
import datetime, random, logging, io, base64, time, os, string, requests
import xgboost as xgb
import numpy as np
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split

# Initialize the Flask application
app = application=Flask(__name__)
# Add Rest API class

#In order for our flask application to communicate with the API Restful functions
api = Api(app)

# global variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = []
xgb_stock_model = None

@app.before_first_request
def PrepareData():
    global FEATURES, xgb_stock_model, options_stocks

    # load model-ready data
    stock_model_ready_df = pd.read_csv(os.path.join(BASE_DIR, 'stock_model_ready.csv'))
    # set all features to strings
    stock_model_ready_df.columns = [str(f) for f in list(stock_model_ready_df)]

    # train the xgb model whenever the server is reset
    # You could also train the model in advance and load the saved version up to skip this step
    xgb_stock_model, FEATURES = GetTrainedModel(stock_model_ready_df, TARGET='outcome')

def GetTrainedModel(stock_model_ready_df, TARGET):
    FEATURES = [f for f in list(stock_model_ready_df) if f not in [TARGET, 'prediction_date', 'last_market_date', 'symbol']]

    x_train, x_test, y_train, y_test = train_test_split(stock_model_ready_df,
                                     stock_model_ready_df[TARGET], test_size=0.33, random_state=42)

    dtrain = xgb.DMatrix(data = x_train[FEATURES],  label = y_train)
    dval = xgb.DMatrix(data = x_test[FEATURES], label = y_test)

    param = {'max_depth':3,
                'eta':0.05,
                'silent':0,
                "objective":"reg:linear",
                "eval_metric":"rmse",
                'subsample': 0.8,
                'maximize': False,
                'colsample_bytree': 0.8}

    evals = [(dtrain,'train'),(dval,'eval')]
    stock_model = xgb.train ( params = param,
                  dtrain = dtrain,
                  num_boost_round = 600,
                  verbose_eval=50,
                  early_stopping_rounds = 500,
                  evals=evals)
    return(stock_model, FEATURES)

def GetLiveStockData(symbol, size=50):
    # we'll use pandas_datareader
    import datetime
    pd.core.common.is_list_like = pd.api.types.is_list_like
    import pandas_datareader.data as web

    try:
        end = datetime.datetime.now()
        start = datetime.datetime.now() - datetime.timedelta(days=60)
        live_stock_data = web.DataReader(symbol, 'yahoo', start, end)
        live_stock_data.reset_index(inplace=True)
        live_stock_data = live_stock_data[[ 'Date', 'Close']]
        # live_stock_data = live_stock_data[['symbol', 'begins_at', 'close_price']]
        # live_stock_data.columns = ['symbol', 'date', 'close']
    except:
        live_stock_data = None

    if (live_stock_data is not None):
        live_stock_data = live_stock_data.sort_values('Date')
        live_stock_data = live_stock_data.tail(size)

        # make data model ready
        live_stock_data['Close'] = pd.to_numeric(live_stock_data['Close'], errors='coerce')
        live_stock_data['Close'] = np.log(live_stock_data['Close'])

        # clean up the data so it aligns with our earlier notation
        #live_stock_data['date'] = pd.to_datetime(live_stock_data['date'], format = '%m/%d/%y')
        # sort by ascending dates as we've done in training
        live_stock_data = live_stock_data.sort_values('Date')

        # build dataset
        X = []

        prediction_dates = []
        last_market_dates = []

        # rolling predictions
        rolling_period = 10
        predict_out_period = 5

        for per in range(rolling_period, len(live_stock_data)):
            X_tmp = []
            for rollper in range(per-rolling_period,per):
                # build the 'features'
                X_tmp += [live_stock_data['Close'].values[rollper]]

            X.append(np.array(X_tmp))

            # add x days to last market date using numpy timedelta64
            prediction_dates.append(live_stock_data['Date'].values[per] + np.timedelta64(predict_out_period,'D'))
            last_market_dates.append(live_stock_data['Date'].values[per])


        live_stock_ready_df = pd.DataFrame(X)
        live_stock_ready_df.columns = [str(f) for f in list(live_stock_ready_df)]

        live_stock_ready_df['prediction_date'] = prediction_dates
        live_stock_ready_df['last_market_date'] = last_market_dates

        live_stock_ready_df.columns = [str(f) for f in live_stock_ready_df]

        return(live_stock_ready_df)
    else:
        return(None)

def GetPrediction(symbol, stock_df, xgb_model, FEATURES):
    # create time-series feature engineering
    doos = xgb.DMatrix(data = stock_df[FEATURES])
    # predictions
    preds = xgb_model.predict(doos)

    # just pull las 10 preds and build chart
    future_df_tmp = stock_df.copy()
    future_df_tmp['forcast'] = np.exp(list(preds))

    # just need a couple of rows
    future_df_tmp = future_df_tmp.tail(20)
    future_df_tmp = future_df_tmp.sort_values('prediction_date')

    return(future_df_tmp)

#DIFFERENCE

#Instead of using a function that displays the results in a html page when using
# restful api's we use a class

#name the class and (you have to pass ) pass resource
class SandP500Forecast(Resource):
  #Here we will build our function within the class function; our function will one value to be passed which is symbol
  def get(self, symbol):
    selected_stock = symbol.upper()
    predicted_price = -999

    #making sure there is some data
    if (len(selected_stock) > 0):
        # get live data of the selected stock
        stock_df = GetLiveStockData(selected_stock)
        if (stock_df is not None):
            if len(stock_df) > 0:
                predicted_price = GetPrediction(selected_stock, stock_df, xgb_stock_model, FEATURES)
                predicted_price =predicted_price['forcast'].values[-1]

            return {selected_stock: str(predicted_price)}
    return {'SandP500Forecast': 'error'}

#calling the class function upon entry 
api.add_resource(SandP500Forecast, '/<string:symbol>')


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80,debug=True)
    # app.run(debug=True)
    