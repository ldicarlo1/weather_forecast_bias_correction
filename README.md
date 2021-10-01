
# Weather Forecast Bias Correction
#### Author: Luca Di Carlo
#### Date: Aug 2021

This small project uses simple machine learning to bias correct temperature, dew point, and 10m wind speed data from the HRRR, GFS, and ECMWF forecasts.

In order to run the jupyter notebook below place the "data/" file containing the station data in this directory.

The contents can be described as the following:

###### analysis.ipynb
A Jupyter Notebook containing the entirety of the assignment, from analyses, modeling, and plot generations.

###### analysis_functions.py
A python script containing the classes of functions I built to use for the analysis.

###### Bias Correction of Weather Forecasts Using Machine Learning.pdf
A slide deck containing a presentation of my findings.



# Results

Results indicated that by using the Gradient Boosted Trees algorithm to bias correct temperature, dew point, and 10m wind speeds produced 
forecasts that were on average 33% more accurate than previously. 

![alt text](https://github.com/ldicarlo1/weather_forecast_bias_correction/blob/main/images/Screen%20Shot%202021-10-01%20at%205.05.42%20PM.png)




