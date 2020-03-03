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

% Run LSTM model with parameters defined in .dat file
hold on
  [fileData,Algorithm_Scheme,choice_training,sampleSize,P_horizon_h,n_points,nVar,XTrain,YTrain]=TestLSTM(fileName,dirCurrent,dirNetwork,dirData);
  cd(dirData) % change directory to the predictions directory
%----------------------------------------------------------------------------------------------
% Estimate prediction skill using RMSE and Jeffreys Divergence metrics
%----------------------------------------------------------------------------------------------
  fileData = strcat(fileData,'_',Algorithm_Scheme,'_',choice_training,'_',num2str(sampleSize),'_',num2str(P_horizon_h),'_', ...
             num2str(n_points),'_',num2str(nVar)); 
         
  load(fileData); % load file with original time series X_true, filtered time series X_filter and predicted X_f

  nVar_f=length(X_f(1,:));
  
  s1=1;
  for i=1:length(t_true)
   for k=1:length(t_f)
     if (t_true(i)==t_f(k))
       rmse(s1,1:nVar_f)=abs(X_f(k,1:nVar_f)-X_true(i,1:nVar_f)); % calculate RMSE for available prediction
       X_original(s1,1:nVar_f)=X_true(i,1:nVar_f); % create a copy of original time series 
       time(s1)=t_true(i); % create a copy of time series
       s1=s1+1;
     end;
   end;
  end;
  lengthT=length(time);
  X1_timeseries=X_original;
  X2_timeseries=X_f;
  hold on
 % Convert from seconds to datetime and to hours format
 %--------------------------------------------------------------------------------------------
 base = datenum(1970,1,1);
 for i=1:length(t_f)
   t_f_time{i}=datestr([t_f(i)]/86400 + base);
 end; 
 for i=1:length(t_true)
   t_true_time{i}=datestr([t_true(i)]/86400 + base);
 end; 
 for i=1:length(t_f)
   t_f_hours(i)=t_f(i)/3600;
 end; 
 for i=1:length(t_true)
   t_true_hours(i)=t_true(i)/3600;
 end; 
 %   Plot results
 %--------------------------------------------------------------------------------------------
 plot(t_true,X_true(:,1),'r','LineWidth',2);
 xticks(t_true(1:200:end));
 xticklabels(t_true_time(1:200:end));
 xtickangle(60);
 plot(t_f(1:5:end),X_f(1:5:end,1),'-mo',...
    'Color',[0,0.7,0.9],'LineWidth',2,...
    'MarkerFaceColor','b',...
    'MarkerSize',4);
 size(X_f)
 %plot(t_f,X_f(:,2),'g');
  %xlimits([t-])
 ylabel('Normalized Oxygen','FontSize',18);
 legend({'Original data';'Prediction'},'FontSize',12);
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