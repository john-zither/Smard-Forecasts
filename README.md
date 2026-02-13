# SMARD Demand Forecast

I thought SMARD's demand forecasts were a little off, and with little effort i could do a decent bit better. To test this, i put together a little script and will be uploading my forecasts regularly.

## Data

I produced 12 different models, starting with limiting myself to the data avalible at noon, ending on the data avalible at midnight. each of these models has validation metrics that outperform by 25% or more, but the more recent models have better performance.

The input data for the model consists only of each regions past demand, and for models after 6 pm, I include the smard forecast values themselves. These models are certianly able to be improved, but this was my first pass.

My forecasts are stored in the forecast_database.csv. The rows are explained below

-  datetime: The time being forecasted
-  time_calculated: The time i ran the script
-  model: The model being used, actual refers to the actual values, smard_forecast refers to the forecasts on the smard website, while the hours 12-24 refer to each of my models
-  value
-  region
