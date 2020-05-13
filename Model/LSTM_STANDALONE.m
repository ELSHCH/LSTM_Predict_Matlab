function [t_f,Y_f_mean,Y_f_std]=LSTM_STANDALONE(time_begin,sampleS,bs,time_end,nVar,t_tr,X_tr,t_down,X_down,ind_predictors,ind_responses,net)
%% Find start and end time indices for filtered time series, 
%  The interval t_filter(ind_point_start:ind_point_end) should overlap t_true(time_start)
%
Y_f=zeros(time_end-time_begin+1,bs,nVar);
Y_f_mean=zeros(time_end-time_begin+1,nVar);
numFeatures = length(ind_predictors);
numResponses = length(ind_responses);
numHiddenUnits = 200;
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
if sampleS==1
 for b_r=1:bs   
%   dt5=[X_down(time_begin,1:nVar)];
%   tt5=[t_down(time_begin)];
  
  dt5=[X_down(1+time_begin-sampleS:time_begin,ind_predictors)]; % set running window of fixed length  
  tt5=[t_down(1+time_begin-sampleS:time_begin)]; % set fixed length time interval
 
% Set initial forecast interval   
  
  t_f=[t_down(time_begin)];
  X_f=[X_down(time_begin,ind_predictors)];
  
%% Reset state of network and update prediction

  %net=resetState(net);
  net = trainNetwork(dt5(1:end-1,ind_predictors)',dt5(2:end,ind_predictors)',layers,options)
% Update prediction using last point of training series    
  [net,YPred] = predictAndUpdateState(net,dt5(end-1,ind_predictors)');
  
  s2=time_begin+1; % set initial iteration index , start right after last training data step
  t_curr=t_down(s2); % set time point to start prediction 
  tt5=[t_curr(1)]; % Update training interval with last prediction
  dt5=[YPred(ind_predictors)'];
  t_f=[t_f,t_curr(1)]; % Insert new time points into prediction time series  
  X_f=[X_f(1:end,ind_predictors); dt5(end,ind_predictors)];
  
 %% Run prediction for 'Nsteps' number of steps 
  
   while s2<time_end
 
      s2=s2+1; % new iteration
      t_curr=t_down(s2); % update time to current time point

% Get next prediction, network is updated with new prediction point 
      [net,YPred] = predictAndUpdateState(net,dt5(end,ind_predictors)','ExecutionEnvironment','cpu');
      tt5=[t_curr(1)]; % update time point 
      dt5=[YPred(ind_predictors)'];   % update input training point

      t_f=[t_f,t_curr]; % insert new time point into prediction interval
      X_f=[X_f(1:end,ind_predictors);dt5(end,ind_predictors)]; % insert new point into prediction interval     
   end;  
   Y_f(:,b_r,ind_predictors)=X_f(:,ind_predictors);
 end;
  for i1=1:length(ind_predictors)
    for i2=1:time_end-time_begin+1  
     Y_f_mean(i2,ind_predictors(i1))=mean(Y_f(i2,:,ind_predictors(i1)));
    end;
  end;  
else   
for b_r=1:bs   
 dt5=[X_down(1+time_begin-sampleS:time_begin,ind_predictors)]; 
 tt5=[t_down(1+time_begin-sampleS:time_begin)];
 
 plot(tt5(end),dt5(end,1),'d','MarkerSize',10);
  
% Set initial forecast interval   

 t_f=[t_down(time_begin)]; 
 X_f=[X_down(time_begin,ind_predictors)];
    
%% Reset state of network and update prediction

   net=resetState(net);
   net = trainNetwork(dt5(1:end-1,ind_predictors)',dt5(2:end,ind_predictors)',layers,options);
% Update prediction using last point of training series    
  [net,YPred] = predictAndUpdateState(net,dt5(end-1,ind_predictors)');

  s2=time_begin+1; % option 1 (training interval) set initial iteration index , start right after last training data step
  t_curr=t_down(s2); % set time point to start prediction 
  tt5=[tt5(2:sampleS),t_curr(1)]; % Update training interval with last prediction
  dt5=[dt5(2:sampleS,ind_predictors);YPred(ind_predictors)'];
  X_f=[X_f(1:end,ind_predictors);dt5(sampleS,ind_predictors)]; % Insert new time points into prediction time series  
  t_f=[t_f,t_curr(1)];    
  plot(t_f(end),X_f(end,1),'d','MarkerSize',17);

%% Run prediction for 'Nsteps' number of steps
    
 while s2<time_end % option 1
      
      s2=s2+1; % new iteration
      t_curr=t_down(s2); % update time to current time point
      
      net = trainNetwork(dt5(1:end-1,ind_predictors)',dt5(2:end,ind_predictors)',layers,options);

 % Get next prediction, network is updated with new prediction point 
      [net,YPred] = predictAndUpdateState(net,dt5(end,ind_predictors)','ExecutionEnvironment','cpu');

      tt5=[tt5(2:sampleS),t_curr(1)]; % shift training time set to the left, new point inserted
      dt5=[dt5(2:sampleS,ind_predictors);YPred(ind_predictors)'];   % shift 

      t_f=[t_f,t_curr]; % insert new time point into prediction interval
      X_f=[X_f(1:end,ind_predictors);dt5(sampleS,ind_predictors)]; % option 1 (training interval)insert new point into prediction interval
%       net = resetState(net);
% % network retrained again with updated interval
% 
%       net = trainNetwork(dt5(1:end-1,1:nVar)',dt5(2:end,1:nVar)',layers,options); % option 1
      
 end;
 Y_f(:,b_r,ind_predictors)=X_f(:,ind_predictors);
 plot(t_f,Y_f(:,b_r,1),'g');
end; 
  for i1=1:length(ind_predictors)
    for i2=1:time_end-time_begin+1  
     Y_f_mean(i2,i1)=mean(Y_f(i2,:,ind_predictors(i1)));
     Y_f_std(i2,i1)=std(Y_f(i2,:,ind_predictors(i1)));
    end;
  end;  
end;