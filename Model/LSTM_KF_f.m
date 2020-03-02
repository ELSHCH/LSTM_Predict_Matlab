function [t_f,Y_f_mean,Y_f_std]=LSTM_KF_f(time_begin,sampleS,bs,time_end,nVar,t_tr,X_tr,t_down,X_down,net)
%% Find start and end time indices for filtered time series, 
%  The interval t_filter(ind_point_start:ind_point_end) should overlap t_true(time_start)
%
Y_f=zeros(time_end-time_begin+1,bs,nVar);
Y_f_mean=zeros(time_end-time_begin+1,nVar);
numFeatures = nVar;
numResponses = nVar;
numHiddenUnits = 100;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    dropoutLayer(0.7)
    lstmLayer(numHiddenUnits)
    dropoutLayer(0.7)
    lstmLayer(numHiddenUnits)
    dropoutLayer(0.7)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('sgdm', ...
    'MaxEpochs',60, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0);% ...
for b_r=1:bs   
 
 dt5=[X_down(1+time_begin-sampleS:time_begin,1:nVar)]; % set running window of fixed length  
 tt5=[t_down(1+time_begin-sampleS:time_begin)]; % set fixed length time interval
 
 % Initialize covariance of state and measurements
 
 Q_init = covarianceStateTransition(dt5,nVar,sampleS,net); % covariance of state
 R_init = sampleCovariance(dt5,nVar,sampleS); % covariance of time series 
 
 Q_std = Q_init; % covariance of state
 R_std = R_init; % covariance of time series 
 Q_f=[Q_std]; 
 R_f=[R_std];
 P = eye(nVar)* 100.; % initialize uncertainty of covariance
 
 s2=time_begin+1; % update time index
 t_curr=t_down(s2); % update time point to start prediction 
 
 % Initialize forecast interval   

 t_f=[t_down(time_begin)]; 
 X_f=[X_down(time_begin,1:nVar)];
 
for tr =1:5 
 % Train network including time series, covariances of model and measurement
 
 net = trainNetwork(dt5(1:end-1,1:nVar)',dt5(2:end,1:nVar)',layers,options); % train network with combined data
  
% % Reset state of network 
% 
%   net=resetState(net);        
%   
  F=eye(nVar);
  H= eye(nVar);
%   output_W = net.IW; % weights of network
%   output_b = net.b; % biases of network
  [net,YPred] = predictAndUpdateState(net,dt5(end,1:nVar)'); % update network state and make step
% forward prediction
% Make estimate of system state and covariances  
%  X  = Ypred*output_w1+output_b1; % estimate new state using network prediction
  X  = YPred; % estimate new state using network prediction

  P = F*P*transpose(F)+Q_std; % initialize uncertainty of covariance
  for i=1:nVar  
   P_d(i) = P(i,i); % initialize uncertainty of covariance
   z(i)=YPred(i)+R_std(i,i);
  end;

  y = z' - H*X(1:nVar);
  P_f=[P_d(1:nVar)]; 
  
 % Estimate Kalman Gain
  K = P*transpose(H)*inv(H*P*transpose(H)+R_std);
 % predict new state with the Kalman Gain correction
  XPred = X + K*y;   
  
% Update prediction using last point of training series    

  tt5=[tt5(2:sampleS),t_curr(1)]; % Update fixed length training interval 
  dt5=[dt5(2:sampleS,1:nVar);XPred(1:nVar)']; % Update running window  
   
  Q_std = covarianceStateTransition(dt5,nVar,sampleS,net); % covariance of state
  R_std = sampleCovariance(dt5,nVar,sampleS); % covariance of time series 
 
  P = (eye(nVar) - K*H)*P*transpose((eye(nVar) - K*H)) + K*R_std*transpose(K)
  for i=1:nVar  
      P_d(i)=P(i,i);
      Q_d(i)=Q_std(i,i);
      R_d(i)=R_std(i,i);
  end;
end;
% Insert new corrected states into prediction time series  
%   P_f=[P_d(1:nVar)];
% 
%   X_f=[X_f(1:end,1:nVar);dt5(sampleS,1:nVar)]; % Insert new time points into prediction time series  
%   t_f=[t_f,t_curr(1)];    
%   Q_f=[Q_d]; 
%   R_f=[R_d];

%% Run prediction for 'Nsteps' number of steps
    
 while s2<time_end % option 1
      
      s2=s2+1; % iterate step forward
      t_curr=t_down(s2); % make tiem step forward
      
 % Get next prediction, network is updated with new prediction point 
  
      [net,YPred] = predictAndUpdateState(net,dt5(end,1:nVar)','ExecutionEnvironment','cpu');
 
%      X  = Ypred*output_w1+output_b1; % estimate new state using network prediction
      X  = YPred; % estimate new state using network prediction
      P = F*P*transpose(F)+Q_std; % initialize uncertainty of covariance
      
      for i=1:nVar  
        P_d(i) = P(i,i); % initialize uncertainty of covariance
        z(i)=YPred(i)+R_std(i,i);
      end;
      y = z' - H*X(1:nVar);
  
 % Estimate Kalman Gain
      K = P*transpose(H)*inv(H*P*transpose(H)+R_std);
 % predict new state with the Kalman Gain correction
      XPred = X(1:nVar) + K*y;   
      for i=1:nVar  
       z(i)=XPred(i)+R_std(i,i); % update new measurement point
      end;
      
      tt5=[tt5(2:sampleS),t_curr(1)]; % shift training time set to the left, new point inserted
      dt5=[dt5(2:sampleS,1:nVar);XPred(1:nVar)']; % Update running window  
      Q_std = covarianceStateTransition(dt5,nVar,sampleS,net); % covariance of state
      R_std = sampleCovariance(dt5,nVar,sampleS); % covariance of time series 
  
      P = (eye(nVar) - K*H)*P*transpose((eye(nVar) - K*H)) + K*R_std*transpose(K);
      for i=1:nVar  
       P_d(i)=P(i,i);
      end;
% Insert new corrected states into prediction time series        
      P_f=[P_f(1:nVar);P_d(1:nVar)];
      t_f=[t_f,t_curr]; % insert new time point into prediction interval
      X_f=[X_f(1:end,1:nVar);dt5(sampleS,1:nVar)]; % option 1 (training interval)insert new point into prediction interval
      Q_f=[Q_f;Q_std];
      R_f=[R_f;R_std];
      
 % network retrained with updated interval     
      net = trainNetwork(dt5(1:end-1,1:nVar)',dt5(2:end,1:nVar)',layers,options);
      
 end;
  Y_f(:,b_r,1:nVar)=X_f(:,1:nVar);
end; 
for i1=1:nVar
    for i2=1:time_end-time_begin+1  
     Y_f_mean(i2,i1)=mean(Y_f(i2,:,i1));
     Y_f_std(i2,i1)=std(Y_f(i2,:,i1));
    end;
end;  