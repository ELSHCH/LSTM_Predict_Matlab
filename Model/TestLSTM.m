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
function [fileData,Algorithm_Scheme,choice_training,sampleSize,P_horizon_h,n_points,option_training,categories_predictors,categories_responses,...
          ind_predictors,ind_responses,nVar,ind_start]=TestLSTM(fileName,dirCurrent,dirNetwork,dirData,numHidden,number_Epochs)
%TestLSTM(fileName,dirCurrent,dirNetwork,dirData)
%clear all
% dirCurrent=pwd;
%  
%  
% fileName= 'InputPrediction.dat'; % file name with user-defined initial parameters for prediction 
  [Algorithm_Scheme,choice_training,time_start,P_horizon_s,n_points,nVar,option_training,sampleSize,fileData]=...
                  InputParam(fileName); % read parameters from initial file
P_horizon_h=P_horizon_s/3600;

% for s7=1:4              
%% Read original data from.mat file
%dirNetwork= '../LSTMNetwork'; % define directory for networks library         
%dirData ='../PredictionData'; % define data directory
cd(dirData); 
% read oceanic observational Data from "NormalizedBoknis.dat"  
fileD=strcat(fileData,'.dat');
fl=fopen(fileD,'r');
%categ=fscanf(fl,'%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n');
if fileD == "GeomarData.dat"
  dd= textscan(fl,'%s\t%s\t%s\t%s\n'); % "GeomarData.dat"
else  
 dd =textscan(fl,'%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'); % "NormalizedBoknis.dat"
end; 
formatOut='yyyy-mm-dd hh:MM:ss';
a = cellstr(dd{1});
X=zeros(length(a)-1,length(dd)-1);
nVar=length(dd)-1;
base = datenum(1970,1,1);
for i=1:length(dd)
a= cellstr(dd{i});
if i > 1
categories{i-1}=string(a(1));
end;
 for j=2:length(a)     
   if i==1
     new_time_sec(j-1)=str2num(string(a(j)));
     n_time=datestr(new_time_sec(j-1)/86400+base, formatOut);
     new_time(j-1,1:19)=n_time(1:19);
   else  
     X(j-1,i-1)=str2num(string(a(j)));
   end;  
 end;
end; 
save fil new_time_sec new_time
file_categories=strcat(dirData,'/ListCategories.dat');
cd(dirCurrent);
% Make selection of categories for prediction variables
[n_Var,categories_predictors,categories_responses,ind_predictors,ind_responses, X_predictors]=selectCategories(file_categories,categories,X,nVar);
%predictor_categories_select={'Air pressure','Oxygen','Pressure GEOMAR','Wind GEOMAR', 'Wind Dir GEOMAR', 'Wind Lighthouse',...
 %                             'Wind Dir Lighthouse','Current East at 2m','Current North at 2m'};
if string(categories_responses{1})=='no categories'
 return;
end; 
nVar=n_Var;
numFeatures=nVar;

[data_X,data_T,date_T,ind_start]=readData(X_predictors,new_time,new_time_sec,time_start,nVar);
if ind_start == 0 || ind_start == -1 
  return;
end;
   %  ind_start=ind_start+(s7-1)*100; 
%plot(new_time,X_predictors(:,4));

%% Prepare time series, standartized time series, define filtered time series and 
%  build predictor and target sequences for training memory network

[t_true,X_true,std_true,mean_true,max_D,t_filter,X_filter,XTrain,YTrain,Nsteps] = ...
       PrepareData(n_points,P_horizon_s,choice_training,sampleSize,nVar,data_X,data_T);
 t_filter_ext=t_filter;
%  plot(t_true,X_true(:,1));
%  hold on;
%  plot(t_filter,X_filter(:,1),'o');
%Find indices for begining and end of prediction in the lower sampled time
%series 'X_filter'
delta_time=new_time_sec(2)-new_time_sec(1);
   for j=1:length(t_filter)-1
      if new_time_sec(ind_start-sampleSize)>=t_filter(j) && new_time_sec(ind_start-sampleSize)<=t_filter(j+1)
          sS_f=j;
      end;
       if new_time_sec(ind_start)>=t_filter(j) && new_time_sec(ind_start)<=t_filter(j+1)
          ind_f_start=j;
      end;
   end; 
if (new_time_sec(end)-new_time_sec(ind_start))<Nsteps*delta_time
   for i=1:Nsteps
     new_time_sec(ind_start+i)= new_time_sec(ind_start)+i*delta_time;
   end; 
end;   
for i=1:Nsteps
 if i*n_points <= Nsteps   
  t_filter_ext(ind_f_start+i)=new_time_sec(ind_start+i*n_points);
 end; 
end;      
for j=1:length(t_filter_ext)-1
   if new_time_sec(ind_start+Nsteps)>=t_filter_ext(j) && new_time_sec(ind_start+Nsteps)<=t_filter_ext(j+1)
      ind_f_end=j;
   end;
end; 
sampleSize_f=ind_f_start-sS_f+1;
lengthF=length(t_filter_ext);
windowSize_f=ind_f_end-ind_f_start+1;
% plot(t_true(ind_start),X_true(ind_start,1),'d','MarkerSize',10);
% plot(t_filter(ind_f_start),X_filter(ind_f_start,1),'d','MarkerSize',10);
% plot(t_true(ind_start-sampleSize),X_true(ind_start-sampleSize,1),'s','MarkerSize',15);
% plot(t_filter(ind_f_start-sampleSize_f),X_filter(ind_f_start-sampleSize_f,1),'s','MarkerSize',15);
% plot(t_true(ind_start+Nsteps),X_true(ind_start+Nsteps,1),'d','MarkerSize',10);
% plot(t_filter(ind_f_end),X_filter(ind_f_end,1),'d','MarkerSize',10);
% Check condition if LSTM pretrained network does already exist
% plot(new_time_sec,new_time_sec,'r');
% hold on
% plot(t_filter(ind_f_start),t_filter(ind_f_start),'s');
% plot(t_filter(ind_f_start-sampleSize_f),t_filter(ind_f_start-sampleSize_f),'o');
% plot(t_filter(ind_f_end),t_filter(ind_f_end),'d');
Files=dir(dirNetwork);
Networksfilename=cell(length(Files)-2,1);
network_exists=0;  % define variable "1" - network already exist in the library of pretrained networks , "0" - does not exist 
network_KF_exists=0;
fileNetwork = strcat(Algorithm_Scheme,'_',choice_training,'_',num2str(sampleSize),'_',num2str(P_horizon_h),'_',num2str(n_points),'_',num2str(nVar),'_',num2str(numHidden),'.mat');
fileNetwork_KF = strcat(Algorithm_Scheme,'_','QRF','_',choice_training,'_',num2str(sampleSize),'_',num2str(P_horizon_h),'_',num2str(n_points),'_',num2str(nVar),'_',num2str(numHidden),'.mat');
for k=1:length(Files)-2
  Networksfilename{k}=Files(k+2).name;
  
  if strcmp(Networksfilename{k},fileNetwork)==1
      network_exists=1;
  end;  
  if strcmp(Networksfilename{k},fileNetwork_KF)==1
      network_KF_exists=1;
      cd(dirNetwork);
      load(Networksfilename{k});
      net_KF=net;
      cd(dirCurrent);
  end;  
end;
num_hidden = numHidden;
if network_exists==0
    % Train LSTM networks "net" for prediction and save the trained network
    switch Algorithm_Scheme
        case "LSTM_PATTERN_KF" 
          numResponses=length(ind_responses);  
          [XTrain,YTrain,net]=prepareTrainData(X_filter(floor(1:length(t_filter)),1:nVar),...
              t_filter(1:floor(length(t_filter))),numFeatures,numResponses,num_hidden,number_Epochs,windowSize_f);
        case "3LSTM_PATTERN_KF"
          numResponses = length(ind_predictors);
          numFeatures = length(ind_predictors);
          % numResponses = 1;
          [XTrain,YTrain,net]=prepareTrainData(X_filter(1:length(t_filter),1:nVar),...
              t_filter(1:floor(length(t_filter))),numFeatures,numResponses,num_hidden,number_Epochs,windowSize_f);
        case "LSTM_PATTERN" 
          numResponses = length(ind_responses);
          numFeatures = length(ind_predictors); 
          [XTrain,YTrain,net]=prepareTrainData(X_filter(floor(1:length(t_filter)),1:nVar),...
              t_filter(1:floor(length(t_filter))),numFeatures,numResponses,num_hidden,number_Epochs,windowSize_f);  
        otherwise
          [net,layers,options] = trainNetworkStep(XTrain,YTrain,nVar); % LSTM network for prediction
    end;
     net_F=net;    
else 
    load(strcat('../LSTMNetwork/',fileNetwork)); % load already pretrained network
    net_F=net;
   % [layers,options]=setParametersNetwork(nVar,num_hidden); % layers and options   
end;  
% Save network if it does not exist yet
if network_exists==0
 saveNetwork(net,dirNetwork,fileNetwork); 
end;
cd(dirCurrent);
%% Folk at Algorithm scheme LSTM STANDALONE / LSTM PATTERN /
%  1 LSTM & KALMAN FILTER / 3 LSTM & KALMAN FILTER
bsize = 1;
%numResponses = numFeatures;
%numResponses = 1;
numHiddenUnits=numHidden;

switch Algorithm_Scheme
    case "LSTM_STANDALONE"
     if network_KF_exists==0 || strcmp(string(option_training),"train")==1   
       [net_F]=LSTM_STANDALONE_train(ind_f_start,sampleSize_f,bsize,ind_f_end,nVar,P_horizon_s,t_true,X_true,t_filter_ext,X_filter,ind_predictors,ind_responses,net_F);
     else   
       [t_f,X_f_mean]=LSTM_STANDALONE(ind_f_start,sampleSize_f,bsize,ind_f_end,nVar,P_horizon_s,t_true,X_true,t_filter_ext,X_filter,ind_predictors,ind_responses,net_F); 
     end;
    case "LSTM_PATTERN_KF"
     if network_KF_exists==0 || strcmp(string(option_training),"train")==1  
       [net_KF,net_F]=LSTM_PATTERN_KF_train(ind_f_start,windowSize_f,bsize,ind_f_end,nVar,P_horizon_s,t_true,X_true,t_filter_ext,X_filter,numHiddenUnits,number_Epochs,ind_predictors,ind_responses,net_F);
     else    
       [t_f,X_f_mean,X_f_std]=LSTM_PATTERN_KF(ind_f_start,windowSize_f,bsize,ind_f_end,nVar,P_horizon_s,t_true,X_true,t_filter_ext,X_filter,numHiddenUnits,number_Epochs,ind_predictors,ind_responses,net_F,net_KF);
     end;
    case "LSTM_PATTERN"
     if network_KF_exists==0 || strcmp(string(option_training),"train")==1        
       [net_KF,net_F]=LSTM_PATTERN_train(ind_f_start,windowSize_f,bsize,ind_f_end,nVar,P_horizon_s,t_true,X_true,t_filter_ext,X_filter,numHiddenUnits,number_Epochs,ind_predictors,ind_responses,net_F); 
     else
       [t_f,X_f_mean,X_f_std]=LSTM_PATTERN(ind_f_start,windowSize_f,bsize,ind_f_end,nVar,P_horizon_s,t_true,X_true,t_filter_ext,X_filter,numHiddenUnits,number_Epochs,ind_predictors,ind_responses,net_F,net_KF);
     end;
    case "3LSTM_PATTERN_KF"
     if network_KF_exists==0 || strcmp(string(option_training),"train")==1  
         'here'
       [net_KF,net_F]=LSTM_PATTERN_KF_QRF_train(ind_f_start,windowSize_f,bsize,ind_f_end,P_horizon_s,t_filter_ext,X_filter,numHiddenUnits,number_Epochs,ind_predictors,ind_responses,net_F); 
     else 
         'here_here'
       [t_f,X_f_mean,X_f_std]=LSTM_PATTERN_KF_QRF_exp(ind_f_start,windowSize_f,bsize,ind_f_end,nVar,P_horizon_s,t_filter_ext,X_filter,numHiddenUnits,number_Epochs,ind_predictors,ind_responses,net_F,net_KF);
     end;
%     case "1LSTM_KF"
%        [t_f,X_f_mean,X_f_std]=LSTM_KF_f(ind_f_start,sampleSize_f,bsize,ind_f_end,nVar,t_true,X_true,t_filter,X_filter,net_F);
%     case "3LSTM_KF"
%        [t_f,X_f_mean,X_f_std]=LSTM_KF_QRFf(ind_f_start,sampleSize_f,bsize,ind_f_end,nVar,t_true,X_true,t_filter,X_filter,net_F);
end;    
% Save data and network in separate higher level directories

fileName = strcat(fileData,'_',Algorithm_Scheme,'_',choice_training,'_',num2str(sampleSize),'_',num2str(P_horizon_h),'_',num2str(n_points),'_',num2str(nVar)); 
dirName = '../PredictionData';
numResponses=length(categories_responses);
for i=1:length(categories_responses)
    cts(i)=categories_responses{i};
end;
% Save prediction if option_training = 'test'
if strcmp(string(option_training),"test")==1 
%    X_f_mean_f=X_f_mean;
%    X_true_f=X_true;
%    X_filter_f=X_filter; 
  [X_f_mean,X_true,X_filter]=un_std(std_true,mean_true,max_D,X_f_mean,X_true,X_filter,ind_responses);  
  savePrediction(t_f,X_f_mean,t_true,X_true,t_filter,X_filter,numResponses,cts,fileName,dirName);
  cd(dirCurrent);
end;  
% Save QRF network if it does not exist yet
if strcmp(string(option_training),"train")==1
 saveNetwork(net_KF,dirNetwork,fileNetwork_KF); 
end;
%% return to current directory
cd(dirCurrent);
% end;
