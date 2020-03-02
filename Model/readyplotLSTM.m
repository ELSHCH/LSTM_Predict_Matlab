function readyplotLSTM(dirCurrent,LSTM_training_f,Algorithm_Scheme_f,choice_training_f,npoints_f)
%-----------------------------------------------------------------------------
%   Estimate and plot Jeffrey's divergence and RMSE from predicted
%   LSTM time series. 
%-----------------------------------------------------------------------------
%    
%   Jeffrey's divergence calculated using function "divergenceJeffrey".
%   RMSE is estimated using function "estimateRMSE"
%   Input parameters: dirCurrent - name of model directory;
%                     LSTM_training - options "POINT","INTERVAL" 
%                     Algorithm_Scheme - options "UPDATE","NOUPDATE","UPDATE_KALMAN"
%                     choice_training - options "FULL","PART"
%                     npoints_f - frequency of low rate sampling; 
%   Last modified Elena 26/11/2019                      
%-------------------------------------------------------------------------%

cd('../PredictionData'); % change directory to the one containing LSTM predictions

FileName=strcat('LSTM',Algorithm_Scheme_f,'_',LSTM_training_f,'_',choice_training_f,'_',num2str(npoints_f));

load(FileName); % load file with original time series X_true, filtered time series X_filter and predicted X_f

nVar_f=length(X_true(1,:));
size(X_f)
size(X_true)
for i=1:length(t_true)
 for k=1:length(X_f)
    if (t_true(i)==t_f(k))
      rmse(k,1:nVar_f)=abs(X_f(k,1:nVar_f)-X_true(i,1:nVar_f)); % calculate RMSE for available prediction
      X_original(k,1:nVar_f)=X_true(i,1:nVar_f); % create a copy of original time series 
      time(k)=t_true(i); % create a copy of time series 
    end;
  end;
end;

% Convert seconds to date
e = datenum('01-01-1970 00:00:00');
for i=1:length(time)
 time_sec(i) = datetime(datestr(e+time(i)/86400,'dd-mmm-yyyy hh:MM:ss'));
end;

lengthT=length(time);
X1_timeseries=X_original;
X2_timeseries=X_f;

cd('../Model'); % change back to current directory 

D_J = divergenceJeffrey(X1_timeseries,X2_timeseries); % calculate Jeffrey's divergence between 
% predicted and original data
subplot(3,1,1);
plot(time,rmse(:,1),'g');
xtickangle(45);
title('Root mean square');
xlim([time(1),time(end)]);
subplot(3,1,2);
plot(time,D_J(:,1));
title('Jeffreys divergence');
xtickangle(45);
subplot(3,1,3);
plot(t_true,X_true(:,1),'b');
hold on
plot(t_f,X_f(:,1),'r');
title('Comparison of true and predicted data');
xtickangle(45)
legend('original','prediction');
