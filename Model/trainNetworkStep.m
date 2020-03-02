function [net,layers,options] = trainNetworkStep(XTrain_f,YTrain_f,nVar)
%  Train LSTM network and set parameters of network
%    Input parameters: XTrain_f - original sequence;
%                      YTrain_f - sequence shifted from original by one step;   
%                      nVar_f - number of time series;
%    Output parameters: net - trained network;
%                       layers, options - network parameters; 
%   Last modified Elena 27/11/2019
%------------------------------------------------------------------------
% Set the parameters for LSTM network
num_hidden=300;
[layers,options]=setParametersNetwork(nVar,num_hidden);

% numFeatures = nVar;
% numResponses = nVar;
% numHiddenUnits = 300;
% 
% layers = [ ...
%     sequenceInputLayer(numFeatures)
%     lstmLayer(numHiddenUnits)
%     dropoutLayer(0.7)
%     lstmLayer(numHiddenUnits)
%     dropoutLayer(0.7)
%     lstmLayer(numHiddenUnits)
%     dropoutLayer(0.7)
%     fullyConnectedLayer(numResponses)
%     regressionLayer];
% 
% options = trainingOptions('adam', ...
%     'MaxEpochs',360, ...
%     'GradientThreshold',1, ...
%     'InitialLearnRate',0.005, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',125, ...
%     'LearnRateDropFactor',0.2, ...
%     'Verbose',0);% ...
% Train LSTM Network ---------------------------------------

 net = trainNetwork(XTrain_f',YTrain_f',layers,options);