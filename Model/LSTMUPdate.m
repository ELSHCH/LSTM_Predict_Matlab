function [t_f,X_f]=LSTMUPdate(time_begin,numpoints,Nsteps,LSTM_option,nVar,t_tr,X_tr,t_down,X_down,net,layers,options)

%% Define training window with fixed length

deltaT = t_tr(2)-t_tr(1);

if LSTM_option == "INTERVAL"
  dt5=[X_tr(1+time_begin-numpoints:time_begin,1:nVar)]; 
  tt5=[t_tr(1+time_begin-numpoints:time_begin)];
  
% Set initial forecast interval   

  t_f=[t_tr(time_begin+1)]; 
  X_f=[X_tr(time_begin+1,1:nVar)];
  
%% Find start and end time indices for filtered time series, 
%  The interval t_filter(ind_point_start:ind_point_end) should overlap t_true(time_start) 

  end_step=time_begin+Nsteps;
  [ind_points_start,ind_points_end]=defineInterval(time_begin,t_tr,t_down,end_step);  
    
%% Reset state of network and update prediction

  net=resetState(net);
% Update prediction using last point of training series    
  [net,YPred] = predictAndUpdateState(net,dt5(end,1:nVar)');

  s2=time_begin+1; % option 1 (training interval) set initial iteration index , start right after last training data step
  t_curr=t_tr(s2); % set time point to start prediction 
  tt5=[tt5(2:numpoints),t_curr(1)]; % Update training interval with last prediction
  dt5=[dt5(2:numpoints,1:nVar);YPred(1:nVar)'];
  X_f=[X_f(1:end,1:nVar);dt5(numpoints,1:nVar)]; % Insert new time points into prediction time series  
  t_f=[t_f,t_curr(1)];    
  

%% Run prediction for 'Nsteps' number of steps
    
 while s2<time_begin+Nsteps % option 1
     
      s2=s2+1; % new iteration
      t_curr=t_tr(s2); % update time to current time point

 % Get next prediction, network is updated with new prediction point 
      [net,YPred] = predictAndUpdateState(net,dt5(end,1:nVar)','ExecutionEnvironment','cpu');

      tt5=[tt5(2:numpoints),t_curr(1)]; % shift training time set to the left, new point inserted
      dt5=[dt5(2:numpoints,1:nVar);YPred(1:nVar)'];   % shift 

      t_f=[t_f,t_curr]; % insert new time point into prediction interval
      X_f=[X_f(1:end,1:nVar);dt5(numpoints,1:nVar)]; % option 1 (training interval)insert new point into prediction interval
      
      for s1=ind_points_start:ind_points_end % loop over filtered time series interval
% check conditions if new update point is reached  
        if (abs(t_down(s1)-t_curr)<=deltaT&&(t_down(s1)<=t_curr)) 
      
          ind_t=s1;
          ind_points_start=ind_t;
% update training input with actual value from original time series
     
          dt5=[X_tr(s2-(numpoints-1):s2,1:nVar)]; 
          tt5=[t_tr(s2-(numpoints-1):s2)];
    
% update points in forecast

          X_f=[X_f(1:end-1,1:nVar);dt5(numpoints,1:nVar)]; % option 1
        
          net = resetState(net);
% network retrained again with updated interval

          net = trainNetwork(dt5(1:end-1,1:nVar)',dt5(2:end,1:nVar)',layers,options); % option 1
        end;  
      end;
end;
    
elseif LSTM_option == "POINT"
  dt5=[X_tr(time_begin,1:nVar)];
  tt5=[t_tr(time_begin)];
 
% Set initial forecast interval   
  
  t_f=[t_tr(time_begin)];
  X_f=[X_tr(time_begin,1:nVar)];

%% Find start and end time indices for filtered time series, 
%  The interval t_filter(ind_point_start:ind_point_end) should overlap t_true(time_start)   
  
  end_step=Nsteps;
  [ind_points_start,ind_points_end]=defineInterval(time_begin,t_tr,t_down,end_step); 
  
%% Reset state of network and update prediction

  net=resetState(net);
% Update prediction using last point of training series    
  [net,YPred] = predictAndUpdateState(net,dt5(end,1:nVar)');
  
  s2=time_begin+1; % set initial iteration index , start right after last training data step
  t_curr=t_tr(s2); % set time point to start prediction 
  tt5=[t_curr(1)]; % Update training interval with last prediction
  dt5=[YPred(1:nVar)'];
  t_f=[t_f,t_curr(1)]; % Insert new time points into prediction time series  
  X_f=[X_f(1:end,1:nVar); dt5(end,1:nVar)];
  
 %% Run prediction for 'Nsteps' number of steps
 
  
   while s2<time_begin+Nsteps
 
      s2=s2+1; % new iteration
      t_curr=t_tr(s2); % update time to current time point

% Get next prediction, network is updated with new prediction point 
      [net,YPred] = predictAndUpdateState(net,dt5(end,1:nVar)','ExecutionEnvironment','cpu');
      tt5=[t_curr(1)]; % update time point 
      dt5=[YPred(1:nVar)'];   % update input training point

      t_f=[t_f,t_curr]; % insert new time point into prediction interval
      X_f=[X_f(1:end,1:nVar);dt5(end,1:nVar)]; % insert new point into prediction interval
      
      
    for s1=ind_points_start:ind_points_end % loop over filtered time series interval
% check conditions if new update point is reached  
     if (abs(t_down(s1)-t_curr)<=deltaT) 
       if (t_down(s1)<=t_curr) 
 
        ind_t=s1;
        ind_points_start=ind_t;
% update training input with actual value from original time
% series
        dt5=[X_tr(s2,1:nVar)]; 
% update points in forecast
        X_f=[X_f(1:end-1,1:nVar);dt5(end,1:nVar)];
        net = resetState(net);
       end; 
     end;
    end;
 end;     
  
end; 