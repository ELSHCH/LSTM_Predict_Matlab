function [t_f,X_f]=LSTMUPdateKF(time_begin,numpoints,Nsteps,LSTM_option,nVar,sampleSize,t_tr,X_tr,t_down,X_down,net,layers,options)
%        Run LSTM prediction with Kalman Filter correction
%  Input parameters: time_begin - initial index of time series for prediction;
%                    numpoints - fixed length of data sequence used for training network;
%                    Nsteps - number of time steps for prediction;  
%                    LSTM_option - option for network training, "INTERVAL"/"POINT" - interval of one data points is used for training correspondinlgy;
%                    nVar - number of time series studied;
%                    sampleSize - size of ensemble for evaluation of prediction mean and covariance;
%                    t_tr,X_tr - original time and data; 
%                    t_down,X_down - downsampled / filtered time series;  
%                    net, layers, options - trained network and its parameters;
%  Output parameters: t_f, X_f - predicted data of length Nsteps, prediction start from X_tr(time_begin+1). 
%                      
%    Last modified Elena 28/11/2019
%---------------------------------------------------------------------------------------------------------------------------------------------------
%% Define training window with fixed length

deltaT = t_tr(2)-t_tr(1);

if LSTM_option == "INTERVAL"  
  dt5=[X_tr(1+time_begin:time_begin+numpoints,1)']; 
  tt5=[t_tr(1+time_begin:time_begin+numpoints)];
  
% Set initial forecast interval   

  t_f=[t_tr(time_begin+numpoints)];
  X_f=[X_tr(time_begin+numpoints,1)'];
%% Find start and end time indices for filtered time series, 
%  The interval t_filter(ind_point_start:ind_point_end) should overlap t_true(time_start) 

  end_step=Nsteps+numpoints;
  [ind_points_start,ind_points_end]=defineInterval(time_begin,t_tr,t_down,end_step);  
    
%% Reset state of network and update prediction

  net=resetState(net);
  
 %% Calculate prediction guess and covariance, covariance of model, covariance of data, Kalman Gain and update prediction point. 
  P_pred = zeros(nVar,nVar);  
  [P_next,Y_next]=algoritmKFUpdate(net,sampleSize,numpoints,time_begin,nVar,dt5(end),P_pred,X_tr);
      
  Y_Pred=Y_next;   % shift control to next point
  P_Pred=P_next;  

  s2=numpoints+time_begin+1; % option 1 (training interval) set initial iteration index , start right after last training data step
  t_curr=t_tr(s2); % set time point to start prediction 
  tt5=[tt5(2:numpoints),t_curr(1)]; % Update training interval with last prediction
  
  dt5=[dt5(2:numpoints),Y_Pred];
  X_f=[X_f,dt5(numpoints)]; % Insert new time points into prediction time series 
  
  t_f=[t_f,t_curr(1)];    

%% Run prediction for 'Nsteps' number of steps
    
  while s2<time_begin+numpoints+Nsteps % option 1
     
      s2=s2+1; % new iteration
      t_curr=t_tr(s2); % update time to current time point

 %% Calculate prediction guess and covariance, covariance of model, covariance of data, Kalman Gain and update prediction point. 
 
      [P_next,Y_next]=algoritmKFUpdate(net,sampleSize,numpoints,time_begin,nVar,Y_Pred,P_pred,X_tr);
      
      Y_Pred=Y_next;   % shift control to next point
      P_Pred=P_next;

      tt5=[tt5(2:numpoints),t_curr(1)]; % shift training time set to the left, new point inserted
     
      dt5=[dt5(2:numpoints),Y_Pred];   % shift 

      t_f=[t_f,t_curr]; % insert new time point into prediction interval
      
      X_f=[X_f,dt5(numpoints)]; % insert new point into prediction interval
      
      for s1=ind_points_start:ind_points_end % loop over filtered time series interval
% check conditions if new update point is reached  
       if (abs(t_down(s1)-t_curr)<=deltaT&&(t_down(s1)<=t_curr)) 
      
          ind_t=s1;
          ind_points_start=ind_t;
% update training input with actual value from original time series
       
          dt5=[X_tr(s2-(numpoints-1):s2,1)']; 
    
          tt5=[t_tr(s2-(numpoints-1):s2)];
    
% update points in forecast
          X_f=[X_f(1:end-1),dt5(numpoints)]; % option 1

          net = resetState(net);
% network retrained again with updated interval
          net = trainNetwork(dt5(1:end-1),dt5(2:end),layers,options); % option 1
        end;  
      end;
end;
    
elseif LSTM_option == "POINT"
  
  dt5=[X_tr(time_begin,1)'];
  tt5=[t_tr(time_begin)];
 
% Set initial forecast interval   
  
  t_f=[t_tr(time_begin)];

  X_f=[X_tr(time_begin,1)'];
 
%% Find start and end time indices for filtered time series, 
%  The interval t_down(ind_point_start:ind_point_end) should overlap t_tr(time_begin,time_begin+end_step)   
  
  end_step=Nsteps;
  [ind_points_start,ind_points_end]=defineInterval(time_begin,t_tr,t_down,end_step); 
  
%% Reset state of network and update prediction

  net=resetState(net); % net is trained already using "FULL" / "PART" training scheme
  
%% Calculate prediction guess and covariance, covariance of model, covariance of data, Kalman Gain and update prediction point.       

  P_pred = zeros(nVar,nVar); 
  [P_next,Y_next]=algoritmKFUpdate(net,sampleSize,numpoints,time_begin,nVar,dt5(end),P_Pred,X_tr);
  
  Y_Pred=Y_next;   % shift control to next point
  P_Pred=P_next;

  s2=time_begin+1; % set initial iteration index , start right after last training data step
  t_curr=t_tr(s2); % set time point to start prediction 
  tt5=[t_curr(1)]; % Update training interval with last prediction
  dt5=[Y_Pred];
  t_f=[t_f,t_curr(1)]; % Insert new time points into prediction time series 
  
  X_f=[X_f(1:end),dt5(end)];

 %% Run prediction for 'Nsteps' number of steps
 
  
 while s2<time_begin+Nsteps
 
      s2=s2+1; % new iteration
      t_curr=t_tr(s2); % update time to current time point

%% Calculate prediction guess and covariance, covariance of model, covariance of data, Kalman Gain and update prediction point.  
  
     [P_next,Y_next]=algoritmKFUpdate(net,sampleSize,numpoints,time_begin,nVar,Y_Pred,P_pred,X_tr);
     
     Y_Pred=Y_next;   % shift control to next point
     P_Pred=P_next;
    
     tt5=[t_curr(1)]; % update time point 
     dt5=[Y_Pred];   % update input training point

     t_f=[t_f,t_curr]; % insert new time point into prediction interval
     
     X_f=[X_f(1:end),dt5(end)]; % insert new point into prediction interval
 
     for s1=ind_points_start:ind_points_end % loop over filtered time series interval
% check conditions if new update point is reached  
      if (abs(t_down(s1)-t_curr)<=deltaT&&(t_down(s1)<=t_curr)) 
 
        ind_t=s1;
        ind_points_start=ind_t;
% update training input with actual value from original time
% series
        dt5=[X_tr(s2,1)]; 
% update points in forecast
        X_f=[X_f(1:end-1),dt5(end)];
        net = resetState(net);
      end;
     end;
  end;     
  
 end; 