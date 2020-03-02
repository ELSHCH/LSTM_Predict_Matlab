function calcplotProbEVENT
dirCurrent='C:/Users/eshchekinova/Documents/BoknisData/LSTMPred/Model';
cd(dirCurrent);
dirData='../PredictionData';
fileName='InPrediction.dat';
[Algorithm_Scheme,choice_training,time_start,P_horizon_s,n_points,nVar,sampleSize,fileData]=...
                  InputParam(fileName);
fileData = strcat(Algorithm_Scheme,'_',choice_training,'_',num2str(sampleSize),'_', ...
             num2str(n_points),'_',num2str(nVar),'.mat'); 
cd(dirData);         
load('NormalizedBoknis.mat');
load(fileData);
s1=1;
nVar=size(X_f,2);
nOb=size(X_f,1);
ind=find(new_time_sec==t_f(1));
%for i=1:length(new_time_sec)
%    for k=1:length(t_f)
%     ind=find(new_time_sec<t_f)
X_s(1:ind,1:nVar)=X(1:ind,1:nVar); % calculate RMSE for available prediction
t_s(1:ind)=new_time_sec(1:ind);
%        t_s(s1)=new_time_sec(i);
%        s1=s1+1
%      else
X_s(ind+1:ind+nOb-1,1:nVar)=X_f(2:nOb,1:nVar); % create a copy of original time series 
t_s(ind+1:ind+nOb-1)=t_f(2:nOb);
%        t_s(s1)=t_f(k); % create a copy of time series
%        s1=s1+1;
%      end;
%    end;
%  end;
 del_t=min(t_f(2)-t_f(1),new_time_sec(2)-new_time_sec(1));
 num=floor((t_s(end)-t_s(1))/del_t);
 t_int=linspace(t_s(1),t_s(end),num);
 for k=1:nVar
  x_int(:,k)=interp1(t_s,X_s(:,k),t_int);
 end;
 plot(t_s(:),X_s(:,1),'k');
%  hold on
%  plot(t_int,x_int(:,1),'o');
X=X_s;
new_time_sec=t_s;
dirGoogleNet='C:/Users/eshchekinova/Documents/BoknisData/GoogleClass';
cd(dirGoogleNet);
save UpdateBoknis X new_time_sec
cd(dirCurrent);
%CreateBoknisData;