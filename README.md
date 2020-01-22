# Building a LSTM Encoder-Deconder using PyTorch to make Sequence-to-Sequence Predictions

## Requirements 
- `Python 3` 
- `PyTorch`
- `numpy`
- `scipy`

## 1 Overview 
There are many instances where we would like to predict how a time series will behave in the future. For example, we may be interested in forcasting web page viewership, weather conditions (temperature, humidity, etc.), power usage, or traffic volume. A sequence-to-sequence prediction for time series data involves using _m_ input values to predict the next _n_ values. An example sequence-to-sequence prediction for the number of views Stephen Hawking's Wikipedia page receives is shown below. Here, the past few months of viewership (black) is used to predict the next month of viewership (red).  

<p align="center">
  <img src="figures/hawking.jpg" width="900">
    <br>
 <em> <font size = "4"> An example time series sequence-to-sequence prediction for the number of views Stephen Hawking's Wikipedia page receives is: given the past few months of viewership, how many times will the page be viewed in the next month? </font> </em>  
</p>

For sequence-to-sequence time series predictions, the past values of the time series often influence future values. In the case of Stephen Hawking's Wikipedia page, XXX. The Long Short-Term Memory (LSTM) neural network is well-suited for this type of problem because it can learn long-term dependencies in the data. To make sequence-to-sequence predictions using a LSTM, we use an encoder-decoder architecture, which is shown below. 

<p align="center">
  <img src="figures/encoder_decoder.jpg" width="700">
    <br>
 <em> <font size = "4"> The LSTM encoder-decoder allows us to make sequence-to-sequence predictions. The LSTM encoder summarizes the information from an the input sequence in an encoded state. The LSTM decoder takes the encoded state and uses it to produce an output sequence.  </font> </em>  
</p>

The LSTM encoder-decoder consists of two LSTMs. The first LSTM, or the encoder, processes an input sequence and generates an encoded state. The encoded state summarizes the information in the input sequence. The second LSTM, or the decoder, uses the encoded state to produce an output sequence. Note that the input and output sequences can have different lengths.  

In this project, we will build a LSTM encoder-decoder to make sequence-to-sequence predictions for time series data. For illustrative purposes, we will apply our model to a synthetic time series dataset. In [Section 2](#2-preparing-the-time-series-dataset), we will prepare the synthetic time series dataset to input into our LSTM encoder-decoder. In [Section 3](#3-build-the-lstm-encoder-decoder-using-pytorch), we will build the LSTM encoder-decoder using PyTorch. In [Section 4](#4-train-the-model-and-make-predictions), we will train the model and use it to make predictions. Finally, in [Section 5](#5-evaluate-lstm-encoder-decoder-on-train-and-test-datasets), we will evaluate our model on the train and test datasets.

## 2 Preparing the Time Series Dataset

For our time series, we consider the noisy sinusoidal curve plotted below. 

<p align="center">
  <img src="code/plots/synthetic_time_series.png" width="900">
  </p>

We treat the first 80 percent of the time series as the training set and the last 20 percent as the test set. The time series, split into the train and test data, is shown below. 

<p align="center">
  <img src="code/plots/train_test_split.png" width="900">
  </p>

Right now, we have our dataset as one long time series. In order to train the LSTM encoder-decoder, we need to subdivide the time series into many shorter sequences of _n<sub>i</sub>_ input values and _n<sub>o</sub>_ target values. We can achieve this by _windowing_ the time series. To do this, we start at the first _y_ value and collect the first _n<sub>i</sub>_ values as input and the next _n<sub>o</sub>_ values as targets. Then, we slide our window to the second (stride = 1) or third (stride = 2) _y_ value and repeat the procedure. An example of the windowing procedure is shown below for _n<sub>i</sub>_ = 3, _n<sub>o</sub>_ = 2, and stride = 1 is shown below. 

<p align="center">
  <img src="figures/windowed_dataset.png" width="700">
  </p>


We organize the input and target values of the windowing process into two matrices. The matrix, _X_, holds the 

We will feed _X_ and _Y_ into our LSTM encoder-decoder for training. and just X for testing


input values: feed into the 

compare the predictions to the 

Let _N_ be the number of times the window fits the data. The LSTM encoder-decoder expects input of shape `[n<sub>]i<\sub>, # times the window fits into the data]`

We have applied the windowing procedure to our synthetic time series, using _n<sub>i</sub>_ = 80, _n<sub>o</sub>_ = 20, and stride = 5. An example is shown below.  

<p align="center">
  <img src="code/plots/windowed_data.png" width="500">
  </p>
  
## 3 Build the LSTM Encoder-Decoder using PyTorch

  We will use PyTorch to build the LSTM encoder-decoder. Although we build the encoder and decoder separate, these two modules work together during the training and when making predictions. 


## 4 Train the Model and Make Predictions

## 5 Evaluate LSTM Encoder-Decoder on Train and Test Datasets

