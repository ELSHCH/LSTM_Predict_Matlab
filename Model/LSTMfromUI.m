%----------------------------------------------------------------------------------
%  Algorithm of prediction of multiple time series using Long-Short Term Memory Networks 
%--------------------------------------------------------------------------------------
%
%    LSTM algorithm is implemented with three schemes : "UPDATE" and
%    "NOUPDATE" and "UPDATE_KALMAN".
%    In scheme with NOUPDATE a standard step-to-step forward prediction by
%    LSTM is used;
%    in the UPDATE scheme the prediction data are updated with
%    original data and the network is retrained using updated prediction;
%    in the UPDATE_KALMAN scheme the prediction data are updated using
%    Kalman gain
%
%    Two options for prediction could be used 1)"INTERVAL"- interval of length
%    "numpoints" could be used for making step forward prediction at each time step
%    2) "POINT" only last data point is used for prediction of next point
%    
%    The network could be trained using entire time series or selected
%    interval, therefore two options for choice of training : 'FULL', 'PART'
%
%    Initial parameters are set in input file 'InputPrediction.dat'
%
%   Last modified Elena 26/11/2019
%-------------------------------------------------------------------------------
%% Read input parameters from .dat file
function LSTMfromUI(fileName,dirCurrent,dirNetwork,dirData)
%TestLSTM(fileName,dirCurrent,dirNetwork,dirData)
%clear all
% dirCurrent=pwd;
%  
%  
% fileName= 'InputPrediction.dat'; % file name with user-defined initial parameters for prediction 
 
[Algorithm_Scheme,choice_training,time_start,P_horizon_s,n_points,nVar,sampleSize,fileData]=...
                  InputParam(fileName); % read parameters from initial file
      
%% Read original data from.mat file
%dirNetwork= '../LSTMNetwork'; % define directory for networks library         
%dirData ='../PredictionData'; % define data directory
cd(dirData); 
fileD=strcat(fileData,'.mat');
load(fileD,'-mat');
cd(dirCurrent);
% Two options are here: Standard Data - "ChickenPoxData.mat", oceanic observational Data - "NormalizedBoknis.mat"  
 if strcmp(fileData,'NormalizedBoknis')==1
  [data_X,data_T,date_T,ind_start]=readData(X,new_time,new_time_sec,time_start,nVar);
 else
  for i=1:length(new_time) 
   for si=1:nVar
     data_X(i,si) = X(i,si);
  end;  
     data_T(i)=new_time(i);
   end;
   ind_start=str2num(time_start);
 end;

%% Prepare time series, standartized time series, define filtered time series and 
%  build predictor and target sequences for training memory network

[t_true,X_true,t_filter,X_filter,XTrain,YTrain,Nsteps] = ...
       PrepareData(n_points,P_horizon_s,choice_training,sampleSize,nVar,data_X,data_T);
   Nsteps
% Find indices for begining and end of prediction in the lower sampled time
% series 'X_filter'
   for j=1:length(t_filter)-1
      if new_time_sec(ind_start+sampleSize)>=t_filter(j) && new_time_sec(ind_start+sampleSize)<=t_filter(j+1)
          ind_f_start=j;
      end;
      if new_time_sec(ind_start+Nsteps)>=t_filter(j) && new_time_sec(ind_start+Nsteps)<=t_filter(j+1)
          ind_f_end=j;
      end;
       if new_time_sec(ind_start)>=t_filter(j) && new_time_sec(ind_start)<=t_filter(j+1)
          sS_f=j;
      end;
   end; 
sampleSize_f=ind_f_start-sS_f;   
if ind_f_end<=ind_f_start
    ind_f_end = ind_f_start+1;
end;    
% Check condition if LSTM pretrained network does already exist
% plot(new_time_sec,new_time_sec,'r');
% hold on
% plot(t_filter(ind_f_start),t_filter(ind_f_start),'s');
% plot(t_filter(ind_f_start-sampleSize_f),t_filter(ind_f_start-sampleSize_f),'o');
% plot(t_filter(ind_f_end),t_filter(ind_f_end),'d');
Files=dir(dirNetwork);
Networksfilename=cell(length(Files)-2,1);
network_exist=0;  % define variable "1" - network already exist in the library of pretrained networks , "0" - does not exist 
for k=1:length(Files)-2
  Networksfilename{k}=Files(k+2).name;
  fileNetwork = strcat(Algorithm_Scheme,'_',choice_training,'_',num2str(sampleSize),'_',num2str(n_points),'_',num2str(nVar),'.mat');
  
  if strcmp(Networksfilename{k},fileNetwork)==1
      network_exist=1;
  end;    
end;
num_hidden = 300;
if network_exist==0
    % Train LSTM networks "net" for prediction and save the trained network
    [net,layers,options] = trainNetworkStep(XTrain,YTrain,nVar); % LSTM network for prediction
    
else 

    load(strcat('../LSTMNetwork/',fileNetwork)); % load already pretrained network
    [layers,options]=setParametersNetwork(nVar,num_hidden); % layers and options
   
end;    

%% Folk at Algorithm scheme LSTM STANDALONE / KALMAN FILTER STANDALONE /
%  1 LSTM & KALMAN FILTER / 3 LSTM & KALMAN FILTER

switch Algorithm_Scheme
    case "LSTM_STANDALONE"
       [t_f,X_f]=LSTM_STANDALONE(ind_f_start,sampleSize_f,ind_f_end,nVar,t_true,X_true,t_filter,X_filter,net); 
    case "KF_STANDALONE"
        [t_f,X_f]=KF_STANDALONE(ind_f_start,sampleSize_f,ind_f_end,LSTM_training,nVar,sampleSize,t_true,X_true,t_filter,X_filter,net,layers,options);
    case "1LSTM_KF_STANDALONE"
        [t_f,X_f]=LSTM_KF_f(ind_f_start,sampleSize_f,ind_f_end,LSTM_training,nVar,sampleSize,t_true,X_true,t_filter,X_filter,net,layers,options);
    case "3LSTM_KF_STANDALONE"
        [t_f,X_f]=LSTM_KF_QRFf(ind_f_start,sampleSize_f,ind_f_end,LSTM_training,nVar,sampleSize,t_true,X_true,t_filter,X_filter,net,layers,options);
end;    

%% Save data and network in separate higher level directories

fileName = strcat(Algorithm_Scheme,'_',choice_training,'_',num2str(sampleSize),'_',num2str(n_points),'_',num2str(nVar)) ;
dirName = '../PredictionData';
 
savePrediction(t_f,X_f,t_true,X_true,t_filter,X_filter,nVar,fileName,dirName);
cd(dirCurrent);

% Save network if it does not exist yet
if network_exist==0
saveNetwork(net,dirNetwork,n_points,fileNetwork,choice_training); 
end;

%% Plot data from LSTM prediction
%cd(dirCurrent)

%readyplotLSTM(dirCurrent,LSTM_training,Algorithm_Scheme,choice_training,n_points);

%% return to current directory
cd(dirCurrent)

