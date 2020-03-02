****************************************************************************                README  FILE
***************************************************************************
*   Instructions for initialization of               
*   prediction algorithm using LSTM network model and 
*   Kalman Filter corrections      
*
*
*   To initialize data use .dat file InPrediction.dat with
*   following parameters:
*   1st line  method                     // choices LSTM_PATTERN,LSTM_PATTERN_KF,3LSTM_PATTERN_KF,LSTM_STANDALONE
*   2nd line  training option            // choice FULL, PART
*   3th line  length of training interval       //    
*   4th line  start time for prediction  // start time in format "2018-07-18 00:00:30" or integer number of steps  
*   5th line  prediction interval        // length of prediction in hours 
*   6th line  number of points           // parameter corresponds to frequency of low rate sampling
*   7th line  number of parameters       // number of parameters for prediction, 1 is a single parameter
*   8th line name of file               // name of file containing time series data
*
*
* USAGE  The simulations are initialized in Matlab by running main file runLSTM.m from command line or LSTMPrediction.mlapp 
*
*
* CONTACT :shchekinova.elena@gmail.com.      