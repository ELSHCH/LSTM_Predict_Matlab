%------------------------------------------------------------------------------------
%   Initializer code for running LSTM model in command line 
%------------------------------------------------------------------------------------
%   Alternatively a Matlab application LSTMPrediction.mlapp is available for running
%   the model from Matlab UI 
% 
%   USAGE: runLSTM
%
%   Last modified E. Shchekinova 09/02/2020
%------------------------------------------------------------------------------------
% Define input parameters and names for running the algorithm

  clear all;

  fileName='InPrediction.dat'; % .dat File containing initial parameters 
  dirCurrent='C:/Users/eshchekinova/Documents/BoknisData/LSTMPred/Model'; % current directory
  dirNetwork='../LSTMNetwork'; % directory for saving trained network
  dirData='../PredictionData'; % directory for saving prediction data
% Define parameters of network  
numHidden= 20;
number_Epochs = 100;
% Run LSTM model with parameters defined in .dat file
hold on
[fileData,Algorithm_Scheme,choice_training,sampleSize,P_horizon_h,n_points,option_training,...
   categories_predictors,categories_responses,ind_predictors,ind_responses,nVar,ind_start]=TestLSTM(fileName,dirCurrent,dirNetwork,dirData,numHidden,number_Epochs);
if string(categories_responses{1})=='no categories'
  return;
end;
if ind_start == 0 
  fprintf('Please select the new time, the start time is too late\n'); 
  return;
elseif ind_start == -1
  fprintf('Please select the new time, the start time is too early\n'); 
  return;    
end;
cd(dirData) % change directory to the predictions directory
if strcmp(string(option_training),"test")==1
%----------------------------------------------------------------------------------------------
% Estimate prediction skill using RMSE and Jeffreys Divergence metrics
%----------------------------------------------------------------------------------------------
  fileData = strcat(fileData,'_',Algorithm_Scheme,'_',choice_training,'_',num2str(sampleSize),'_',num2str(P_horizon_h),'_', ...
             num2str(n_points),'_',num2str(nVar)); 
         
  load(fileData); % load file with original time series X_true, filtered time series X_filter and predicted X_f
%plot(t_true,X_true(:,1),'r');


  nVar_f=length(X_f(1,:));
  
  s1=1;


%   for i=1:length(t_true)
%    for k=1:length(t_f)-1
%      if (t_true(i)>=t_f(k))&&(t_true(i)<t_f(k+1))
%        rmse(s1,1:nVar_f)=abs(X_f(k,1:nVar_f)-X_true(i,ind_responses(1:nVar_f))) % calculate RMSE for available prediction
%        X_original(s1,1:nVar_f)=X_true(i,ind_responses(1:nVar_f)); % create a copy of original time series 
%        time(s1)=t_f(k); % create a copy of time series
%        s1=s1+1;
%      end;
%    end;
%   end;
%   lengthT=length(time);
%   X1_timeseries=X_original;
%   X2_timeseries=X_f;
%   hold on
 % Convert from seconds to datetime and to hours format
 %--------------------------------------------------------------------------------------------
 base = datenum(1970,1,1);
 for i=1:length(t_filter)
   t_filter_time{i}=datestr([t_filter(i)]/86400 + base);
   t_total_time{i}=datestr([t_filter(i)]/86400 + base);
   t_total(i)=t_filter(i);
 end; 
 for i=1:length(t_f)
   t_f_hours(i)=t_f(i)/3600;
 end; 
 for i=1:length(t_filter)
   t_true_hours(i)=t_filter(i)/3600;
 end; 
 for i=1:length(t_f)
   t_f_time{i}=datestr([t_f(i)]/86400 + base);
   t_total_time{length(t_filter)+i}=datestr([t_f(i)]/86400 + base);
   t_total(length(t_filter)+i)=t_f(i);
 end; 
 %   Plot results
 %--------------------------------------------------------------------------------------------
 num= ind_responses(1); % number of predicted category from parameter list ( see file ListCategories.dat)
 plot(t_filter,X_filter(:,num),'r','LineWidth',2);
 hold on
% plot(t_filter,X_filter(:,num),'o')
% 't_filter'
% length(t_filter)
%  xticks(t_true(1:200:end));
%  xticklabels(t_true_time(1:200:end));
%  xtickangle(60);
 plot(t_f(1:end),X_f(1:end,1),'-mo',...
    'Color',[0,0.7,0.9],'LineWidth',2,...
    'MarkerFaceColor','b',...
    'MarkerSize',4);
 xticks(t_total(1:100:end));
 xticklabels(t_total_time(1:100:end));
 xtickangle(60);
 xlim([t_total(1),t_total(end)]);

% plot(time(1:end),X_original(1:end,2),'-s',...
%     'Color',[0.1,0.9,0.9],'LineWidth',2,...
%     'MarkerFaceColor','b',...
%     'MarkerSize',4);
% plot(t_f,X_f(:,2),'g');
%  xlimits([t-])
 ylabel(categories_predictors{num},'FontSize',18);
 legend({'Original data';'Prediction'},'FontSize',12); 
 end;
cd(dirCurrent); % change back to current directory 
% %   D_J = divergenceJeffrey(X1_timeseries,X2_timeseries); % calculate Jeffrey's divergence for original and predicted data 
% %   for i=1:nVar_f
% %     mean_rmse(i)=mean(rmse(:,i));
% %     mean_DJ(i)=mean(D_J(:,i));
% %   end;
% %   text(time(10),X_original(10,1),num2str(mean_rmse(1)));
% %   text(time(20),X_original(10,1),num2str(mean_DJ(1)));
% % %-------------------------------------------------------------------------------------
% % % Convert seconds to date
% % %-------------------------------------------------------------------------------------
% %   e = datenum('01-01-1970 00:00:00');
% %   for i=1:length(time)
% %    time_date(i) = datetime(datestr(e+time(i)/86400,'dd-mmm-yyyy hh:MM:ss'));
% %   end;
% % %--------------------------------------------------------------------------------------
% % % Graphical output of data: original X_original and predictions X_f              
% % %--------------------------------------------------------------------------------------
% %   plot(time,X_original(:,1),'r',time,X_f(:,1),'g');
% %   xtickangle(45);
% %   %xlabel(app.UIAxes3,time_date);
% %   title('Original and predicted data');
% %   xlim([time(1),time(end)]);
% %   legend('original','prediction');
% % %--------------------------------------------------------------------------------------