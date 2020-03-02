function [t_f,Y_f_mean,Y_f_std]=LSTM_KF_QRFf(time_begin,sampleS,bs,time_end,nVar,t_tr,X_tr,t_down,X_down,net)
%% Find start and end time indices for filtered time series, 
%  The interval t_filter(ind_point_start:ind_point_end) should overlap t_true(time_start)
%
Y_f=zeros(time_end-time_begin+1,bs,nVar);
Y_f_mean=zeros(time_end-time_begin+1,nVar);
numFeatures = 3*nVar;
numResponses = 3*nVar;
numHiddenUnits = 300;
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
 t_f=0;
 Y_f_mean=0;
 Y_f_std=0;
 Q_std = Q_init; % covariance of state
 R_std = R_init; % covariance of time series 
 
 for i=1:nVar
  R_d(1:sampleS,i)=R_std(i,i);
  Q_d(1:sampleS,i)=Q_std(i,i);
 end;

 X_f=[dt5(sampleS,1:nVar)]; % Insert new time points into prediction time series  
 t_f=[t_down(time_begin)];    
 P = 0; % initialize uncertainty of covariance
for tr=1:10
 % Train network including time series, covariances of model and measurement

 zt=[dt5,Q_d,R_d]; % set running window with combined data
 

 net_QRF = trainNetwork(zt(1:end-1,1:3*nVar)',zt(2:end,1:3*nVar)',layers,options); % train network with combined data
 
  F=eye(nVar);
  H= eye(nVar);
%   output_W = net_QRF.IW; % weights of network
%   output_b = net_QRF.b; % biases of network
  [net_QRF,YPred] = predictAndUpdateState(net_QRF,zt(end,1:3*nVar)'); % update network state and make step
 % forward prediction
 % Make estimate of system state and covariances  
%  X  = Ypred*output_w1+output_b1; % estimate new state using network prediction
  X  = YPred; % estimate new state using network prediction  
  Q_std=eye(nVar).*X(nVar+1:2*nVar); % update model covariance
  R_std=eye(nVar).*X(2*nVar+1:3*nVar); % update covariance of measurements
  tr
  P = F*P*transpose(F) +Q_std % initialize uncertainty of covariance
  
  for i=1:nVar  
   P_d(i) = P(i,i); % initialize uncertainty of covariance
   z(i)=YPred(i)+R_std(i,i);
   Q_d(i)=Q_std(i,i);
   R_d(i)=R_std(i,i);
  end;
 
  y = z' - H*X(1:nVar);
  
 % Estimate Kalman Gain
 % K = P*transpose(H)*inv(H*P*transpose(H)+R_std);
 K = P*transpose(H)*inv(H*P*transpose(H)+R_std);

 % predict new state with the Kalman Gain correction
  XPred = X(1:nVar) + K*y;   
  
% Update prediction using last point of training series    

  s2=time_begin+1; % update time index
  t_curr=t_down(s2); % update time point to start prediction 
  tt5=[tt5(2:sampleS),t_curr(1)]; % Update fixed length training interval 
  dt5=[dt5(2:sampleS,1:nVar);XPred(1:nVar)']; % Update running window  
 
  P = (eye(nVar) - K*H)*P*transpose((eye(nVar) - K*H)) + K*R_std*transpose(K);
  %P = (eye(nVar) - K*H)*P;
  for i=1:nVar
    P_d(i)=P(i,i);
  end;   
  % Reset state of network 

  %net_QRF=resetState(net_QRF);
end;  

% Insert new corrected states into prediction time series  
  P_f=[P_d(1:nVar)];
  X_f=[X_f(1:end,1:nVar);dt5(sampleS,1:nVar)]; % Insert new time points into prediction time series  
  t_f=[t_f,t_curr(1)];    
  Q_f=[Q_d]; 
  R_f=[R_d];

%% Run prediction for 'Nsteps' number of steps
    
 while s2<time_end % option 1
      
      s2=s2+1; % iterate step forward
      t_curr=t_down(s2); % make tiem step forward
      
 % Get next prediction, network is updated with new prediction point 
  
      [net_QRF,YPred] = predictAndUpdateState(net_QRF,zt(end,1:3*nVar)','ExecutionEnvironment','cpu');
 
 %     X  = Ypred*output_w1+output_b1; % estimate new state using network prediction
      X  = YPred; % estimate new state using network prediction
      Q_std=eye(nVar).*X(nVar+1:2*nVar); % update model covariance
      R_std=eye(nVar).*X(2*nVar+1:3*nVar); % update covariance of measurements
      P = F*P*transpose(F) +Q_std; 
      for i=1:nVar  
        P_d(i) = P(i,i); % initialize uncertainty of covariance
        z(i)=YPred(i)+R_std(i,i); % update new measurement point
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
      P = (eye(nVar)- K*H)*P*transpose((eye(nVar)- K*H)) + K*R_std*transpose(K);
%       zt=[dt5,Q_d,R_d];
%       Q_std = covarianceStateTransition(dt5,nVar,sampleS,net); % covariance of state
%       R_std = sampleCovariance(dt5,nVar,sampleS); % covariance of time series 
      
      for i=1:nVar 
       Q_d(i)=Q_std(i,i);  
       P_d(i)=P(i,i);
       R_d(i)=R_std(i,i);
      end;    
      zt=[dt5,Q_d,R_d]; % Update full system state
      
% Insert new corrected states into prediction time series        
      P_f=[P_f(1:nVar);P_d(1:nVar)];
      t_f=[t_f,t_curr]; % insert new time point into prediction interval
      X_f=[X_f(1:end,1:nVar);dt5(sampleS,1:nVar)]; % option 1 (training interval)insert new point into prediction interval
      Q_f=[Q_f(1:nVar);Q_d];
      R_f=[R_f(1:nVar);R_d];
      
      
 % network retrained with updated interval     
  %    net_QRF = trainNetwork(zt(1:end-1,1:3*nVar)',zt(2:end,1:3*nVar)',layers,options);
      
 end;
  Y_f(:,b_r,1:nVar)=X_f(:,1:nVar);
  plot(t_f,Y_f(:,b_r,1),'g');
 end; 
for i1=1:nVar
    for i2=1:time_end-time_begin+1  
     Y_f_mean(i2,i1)=mean(Y_f(i2,:,i1));
     Y_f_std(i2,i1)=std(Y_f(i2,:,i1));
    end;
end;  
