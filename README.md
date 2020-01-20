# Building a LSTM Encoder-Deconder using PyTorch to make Sequence-to-Sequence Predictions

## Requirements 
- Python 3 
- PyTorch

## 1 Overview 
There are many instances where we want to predict how a time series will behave in the future. For example, given the number of times a web page was visited in the past, we may want to predict how many many times the page will be viewed in the future.

<p align="center">
  <img src="figures/hawking.jpg" width="900">
    <br>
 <em> <font size = "4"> Forcasting web page traffic is a time series prediction problem. Using past viewership, can we predict how many times Stephen Hawking's Wikipedia page will be viewed in the future? </font> </em>  
</p>

Other examples of time series that we may wish to predict future values of include weather conditions (temperature, humidity, etc.), power usage, and traffic volume. The Long Short-Term Memory (LSTM) neural network is well-suited to these problem since the data may have long-term dependencies (i.e., past values may influence future values). 

In this project, we make a sequence-to-sequence predictions, or prediction where the input and output sequences might be different lengths, using a LSTM. For Stephen Hawking's web page traffic, this might involve providing the LSTM with 20 days of past viewership and predicting the next 5 days. To make sequence-to-sequence predictions, we use a LSTM with a special architecture: the LSTM encoder-decoder. 
