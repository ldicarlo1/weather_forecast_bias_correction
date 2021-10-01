import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

class Preprocessing():
    '''This class of functions contains the necessary functions to preprocess the analysis data.
    '''

    def calc_windspeed(dataframe):
        '''Function inputs a dataframe and calculates the wind speed from u/v components. Returns updated dataframe with
        wind_10m and removes the original u/v components.
        
        Args:
            dataframe (DataFrame-obj) : Input dataframe with variables "v_wind_10m" & "v_wind_10m"
            
        Output:
            dataframe (DataFrame-obj) : Outputs dataframe with "wind_10m" and removes variables "v_wind_10m" & "v_wind_10m"
        
        '''
        v10 = dataframe.v_wind_10m
        u10 = dataframe.u_wind_10m
        
        # calculate wind speed
        w10 = np.sqrt(u10**2 + v10**2)
        
        dataframe["wind_10m"] = w10
        dataframe = dataframe.drop(['u_wind_10m', 'v_wind_10m'], axis=1)
        
        return dataframe

    def merge_forecast_with_obs(forecast_df, obs_df,max_fcst_dt=None):
        '''This function creates a dataframe so each forecast hour has a corresponding observation hour for easy verification.
        Also combines station data.
        
        Args:
            forecast_df (list of DataFrames) : List of forecast DataFrames for each location.
            
            obs_df (list of DataFrames) : List of observation DataFrames for each location.
            
            max_fcst_dt (int) : Largest number of forecast hours to index by. Default uses all forecast hours.
        '''
        dfs=[]
        for fcst_ind in range(len(forecast_df)):
            dfs.append(pd.merge(forecast_df[fcst_ind],obs_df[fcst_ind],left_index=True, right_index=True))
            
        # merging the two dataframe on the DateTime Index removes any excess forecast hours that do not correpond to observational hours
        merged_df = pd.concat(dfs)
        
        if max_fcst_dt == None:
            pass
        else:
            # specify the max fcst_dt hour, removes all fcst_dt's greater than this hour
            merged_df = merged_df[merged_df.fcst_dt<=max_fcst_dt]
        
        
        return merged_df

    def calc_rmse_by_station(analysis_df,model):
        '''Calculate RMSE for each dataframe of model forecast and observation hours by location. 
        
        Args:
            analysis_df (DataFrame - obj) : Dataframe containing the model forecast and predictions.
            
            model (str) : The name of the model being input.
        
        Output:
            (DataFrame) : A dataframe containing the RMSE values by each location in addition to the model used.
        '''
            
        # calculate the error for each location
        KBOS_rmse_temp = mean_squared_error(y_pred=analysis_df[analysis_df.pt=="KBOS"].sfc_air_temperature,y_true=analysis_df[analysis_df.pt=="KBOS"].temperature_c,squared=False)
        KHLC_rmse_temp = mean_squared_error(y_pred=analysis_df[analysis_df.pt=="KHLC"].sfc_air_temperature,y_true=analysis_df[analysis_df.pt=="KHLC"].temperature_c,squared=False)
        KBOS_rmse_dpt = mean_squared_error(y_pred=analysis_df[analysis_df.pt=="KBOS"].sfc_dewpoint,y_true=analysis_df[analysis_df.pt=="KBOS"].dewpoint_c,squared=False)
        KHLC_rmse_dpt = mean_squared_error(y_pred=analysis_df[analysis_df.pt=="KHLC"].sfc_dewpoint,y_true=analysis_df[analysis_df.pt=="KHLC"].dewpoint_c,squared=False)
        KBOS_rmse_w10 = mean_squared_error(y_pred=analysis_df[analysis_df.pt=="KBOS"].wind_10m,y_true=analysis_df[analysis_df.pt=="KBOS"].wind_speed_ms,squared=False)
        KHLC_rmse_w10 = mean_squared_error(y_pred=analysis_df[analysis_df.pt=="KHLC"].wind_10m,y_true=analysis_df[analysis_df.pt=="KHLC"].wind_speed_ms,squared=False)
        
        # calculate mean performance RMSE
        KBOS_mean_rmse = np.mean(KBOS_rmse_temp+KBOS_rmse_dpt+KBOS_rmse_w10)
        KHLC_mean_rmse = np.mean(KHLC_rmse_temp+KHLC_rmse_dpt+KHLC_rmse_w10)
        
        location_rmse = pd.DataFrame({"location":["KBOS","KHLC"],"RMSE":[KBOS_mean_rmse,KHLC_mean_rmse],"model":[model,model]})
        
        return location_rmse

    def calc_rmse_by_variable(analysis_df,model):
        '''Calculate RMSE for each dataframe of model forecast and observation hours by variable. 
        
        Args:
            analysis_df (DataFrame - obj) : Dataframe containing the model forecast and predictions.
            
            model (str) : The name of the model being input.
        
        Output:
            (DataFrame) : A dataframe containing the RMSE values by each variable in addition to the model used.
        '''
            
        # calculate the error for each variable
        rmse_temp = mean_squared_error(y_pred=analysis_df.sfc_air_temperature,y_true=analysis_df.temperature_c,squared=False)
        rmse_dpt = mean_squared_error(y_pred=analysis_df.sfc_dewpoint,y_true=analysis_df.dewpoint_c,squared=False)
        rmse_w10 = mean_squared_error(y_pred=analysis_df.wind_10m,y_true=analysis_df.wind_speed_ms,squared=False)
        
        
        location_rmse = pd.DataFrame({"variable":["temp","dpt","w10"],"RMSE":[rmse_temp,rmse_dpt,rmse_w10],"model":[model,model,model]})
        
        return location_rmse       
    

class Bias_Correction():
    '''This class of functions contains the functions to bias-correct NWP model forecasts.
    '''


    def gradient_boosted_trees_bias_correction(analysis_df,station,model):
        '''This function uses Gradient Boosted Trees by sci-kit learn to bias correct temp, dpt,
        and wind forecast data. This also produces the previous "raw" forecast RMSE and the "bias-corrected"
        corrected" forecast RMSE for comparison.
        
        Args:
            analysis_df (DataFrame-obj) : The dataframe containing the raw forecast & observational data.
            
            station (str) : A string of the station data that is being input.
            
            model (str) : A string of the model data that is being input.
            
        Output:
            (DataFrame-obj) : A dataframe containing the RMSE of the raw and bias-corrected forecasts.
        '''
        
        # use fcst_dt and model forecasted dewpoints as inputs to the model, and the obs data as the target
        X_dpt = analysis_df[["fcst_dt","sfc_dewpoint"]]
        Y_dpt = analysis_df["dewpoint_c"]
        X_temp = analysis_df[["fcst_dt","sfc_air_temperature"]]
        Y_temp = analysis_df["temperature_c"]
        X_w10 = analysis_df[["fcst_dt","wind_speed_ms"]]
        Y_w10 = analysis_df["wind_10m"]

        # split into training/testing use 80/20 split
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=42)
        X_train_dpt, X_test_dpt, y_train_dpt, y_test_dpt = train_test_split(X_dpt, Y_dpt, test_size=0.2, random_state=42)
        X_train_w10, X_test_w10, y_train_w10, y_test_w10 = train_test_split(X_w10, Y_w10, test_size=0.2, random_state=42)

        
        # train a Gradient Boosted Tree algorithm and print the results
        gbr_temp = GradientBoostingRegressor(n_estimators=40)
        gbr_temp.fit(X_train_temp,y_train_temp)
        pred_temp = gbr_temp.predict(X_test_temp)
        
        # collect metrics for temp
        rmse_temp = mean_squared_error(y_test_temp,pred_temp,squared=False)
        rmse_old_temp = mean_squared_error(y_test_temp,X_test_temp.sfc_air_temperature,squared=False)
        
        # train dewpoint
        gbr_dpt = GradientBoostingRegressor(n_estimators=40)
        gbr_dpt.fit(X_train_dpt,y_train_dpt)
        pred_dpt = gbr_dpt.predict(X_test_dpt)
        
        # collect metrics for r2
        rmse_dpt = mean_squared_error(y_test_dpt,pred_dpt,squared=False)
        rmse_old_dpt = mean_squared_error(y_test_dpt,X_test_dpt.sfc_dewpoint,squared=False)
        
        # train w10
        gbr_w10 = GradientBoostingRegressor(n_estimators=40)
        gbr_w10.fit(X_train_w10,y_train_w10)
        pred_w10 = gbr_w10.predict(X_test_w10)
        
        # collect metrics for r2
        rmse_old_w10 = mean_squared_error(y_test_w10,X_test_w10.wind_speed_ms,squared=False)
        rmse_w10 = mean_squared_error(y_test_w10,pred_w10,squared=False)
        
        locations = [station,station,station,station,station,station]
        models = [model,model,model,model,model,model]
        variables = ["temp","dpt","w10","temp","dpt","w10"]
        rmse = [rmse_temp,rmse_dpt,rmse_w10,rmse_old_temp,rmse_old_dpt,rmse_old_w10]
        method = ["bias-corrected","bias-corrected","bias-corrected","raw forecast","raw forecast","raw forecast"]
        
        results_df = pd.DataFrame({"variable":variables,"rmse":rmse,"method":method,"model":models,"location":station})
        
        return results_df
    
        