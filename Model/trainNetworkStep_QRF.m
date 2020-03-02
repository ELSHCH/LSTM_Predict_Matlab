function [net_QRF,layers,options] = trainNetworkStep_QRF(XTrain_f,YTrain_f,nVar,net)
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
[layers,options]=setParametersNetwork(3*nVar,num_hidden);

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
 
 % Initialize covariance of state and measurements
 X=[XTrain_f;YTrain_f(end,1:nVar)];
 R_d=zeros(size(X,1),nVar);
 Q_d=zeros(size(X,1),nVar);

 Y(1:size(X,1),1:nVar)=X(1:size(X,1),1:nVar); 
 Q_std = covarianceStateTransition(Y,nVar,size(X,1),net) % covariance of state
  R_std = sampleCovariance(Y,nVar,3); % covariance of time series 
  for j=1:size(X,1)
%      j
%      size(X,1)
 % Y(1:size(X,1),1:nVar)=X(j-10000:j+9999,1:nVar);   
%   Q_std = covarianceStateTransition(Y,nVar,size(X,1),net); % covariance of state
%   R_std = sampleCovariance(Y,nVar,3); % covariance of time series 
   for i=1:nVar
    R_d(j,i)=R_std(i,i);
    Q_d(j,i)=Q_std(i,i);
   end;
end;
%  for i=1:nVar
%      for k=1:20000
%       R_d(k,i)=R_d(10002,i);
%       Q_d(k,i)=Q_d(10002,i);
%      end;
%      for k=size(X,1)-20000:size(X,1)
%       R_d(k,i)=R_d(size(X,1)-9999,i);
%       Q_d(k,i)=Q_d(size(X,1)-9999,i);
%      end;
%  end;
%  size(R_d)
%  size(Q_d)
plot(1:size(X,1),R_d(:,1)); 
 % Train network including time series, covariances of model and measurement

 zt=[X,Q_d,R_d]; % set running window with combined data
 size(zt)
 net_QRF = trainNetwork(zt(1:end-1,1:3*nVar)',zt(2:end,1:3*nVar)',layers,options); % train network with combined data