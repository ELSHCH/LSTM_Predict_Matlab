clear all
fileName='InPrediction.dat'; % .dat File containing initial parameters 
fileManualInput = 'InPredictionManual.dat'; % .dat File with initial parameters written inside the iteration loop
dirCurrent='C:/Users/eshchekinova/Documents/BoknisData/LSTMPred/Model'; % current directory
dirNetwork='../LSTMNetwork'; % directory for saving trained network
dirData='../PredictionData'; % directory for saving prediction data
% Define parameters of network
numHidden = [2:2:20];
%number_Epochs=10;
number_Epochs=100;
% Set parameters of simulation
Pred_horizion=12;
init_time_date='10-Aug-2018';
init_time_sec=posixtime(datetime(init_time_date,'Format','dd-MMM-yyyy'));

for i=1:20
  time_start(i)=init_time_sec+24*3600*(i-1);
end;
size(time_start)
base = datenum(1970,1,1);
 for i=1:length(time_start)
   t_f_time{i}=datestr([time_start(i)]/86400 + base);
   dd=char(string(t_f_time{i}));
   ddi(i,1:11)=dd;
 end; 

X_orig={ };
X_2={ };
for si1=1:20
    si1
fin=fopen(fileManualInput,'w');

fprintf(fin,'%s\n','LSTM_PATTERN'); % three types of schemes are used 'LSTM STANDALONE',...
            % 'KALMAN FILTER STANDALONE','1 LSTM & KALMAN FILTER' and '3 LSTM & KALMAN FILTER'

fprintf(fin,'%s\n','FULL'); % two choices are used 'FULL','PART'

fprintf(fin,'%d\n',100); % length of training interval if 'INTERVAL' training is chosen

fprintf(fin,'%s\n',ddi(si1,1:11)); %start time in format "2018-07-18 00:00:30" for prediction     

fprintf(fin,'%d\n',12); % prediction horizon (in hours);
   
fprintf(fin,'%d\n',2); % frequency for low rate sampling/averaging of data;    

fprintf(fin,'%d\n',12); % number of parameters     

fprintf(fin,'%s\n','NormalizedBoknis'); % name of data file   

fclose(fin);
userInput = 0;
for si3=1:10
    si3
for si2=1:1
    si2
if userInput == 1
    
    [fileData,Algorithm_Scheme,choice_training,sampleSize,P_horizon_h,n_points,nVar]=TestLSTM(fileName,dirCurrent,dirNetwork,dirData,numHidden(si3),number_Epochs);
else
    [fileData,Algorithm_Scheme,choice_training,sampleSize,P_horizon_h,n_points,nVar]=TestLSTM(fileManualInput,dirCurrent,dirNetwork,dirData,numHidden(si3),number_Epochs);
end;    
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
       rmse(si1,s1,1:nVar_f)=abs(X_f(k,1:nVar_f)-X_true(i,1:nVar_f)); % calculate RMSE for available prediction
       X_original(s1,1:nVar_f)=X_true(i,1:nVar_f); % create a copy of original time series 
       X2(s1,1:nVar_f)=X_f(k,1:nVar_f);
       time(s1)=t_f(k); % create a copy of time series
       s1=s1+1;
     end;
   end;
  end;
  cd(dirCurrent);
%   D_J = divergenceJeffrey(X_original,X2);
%   dj(si1,si2)=mean(D_J(:,1));
  mean_rmse(si1,si2,si3)=mean(rmse(si1,:,1));
%   X_orig={X_orig; X_original};
%   X_2={X_2;X2};
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
%  plot(t_f(1:5:end),X_f(1:5:end,1),'-mo',...
%     'Color',[0,0.7,0.9],'LineWidth',2,...
%     'MarkerFaceColor','b',...
%     'MarkerSize',4);

plot(time(1:5:end),X_original(1:5:end,1),'-s',...
    'Color',[0.1,0.9,0.9],'LineWidth',2,...
    'MarkerFaceColor','b',...
    'MarkerSize',4);
plot(time(1:5:end),X2(1:5:end,1),'-s',...
    'Color',[1,0.9,0.9],'LineWidth',2,...
    'MarkerFaceColor','b',...
    'MarkerSize',4);

 %plot(t_f,X_f(:,2),'g');
  %xlimits([t-])
 ylabel('Normalized Oxygen','FontSize',18);
 legend({'Original data';'Prediction'},'FontSize',12);
cd(dirCurrent); % change back to current directory 
clearvars X_original X2 time
end;
end;
end;
save RMSE_Sim_LSTM_100_12 mean_rmse 
%    D_J = divergenceJeffrey(X1_timeseries,X2_timeseries); % calculate Jeffrey's divergence for original and predicted data 
%   for i=1:nVar_f
%     mean_rmse(i)=mean(rmse(:,i));
%     mean_DJ(i)=mean(D_J(:,i));
%   end;
%   text(time(10),X_original(10,1),num2str(mean_rmse(1)));
%   text(time(20),X_original(10,1),num2str(mean_DJ(1)));