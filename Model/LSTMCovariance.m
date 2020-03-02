function Matrix_covar_LSTM= ...
                   LSTMCovariance(t_f,y_f)
deltaT=t_f(2)-t_f(1);
numD=length(X_downscale);
len_d=length(y_f);

%data = [data{:}];
%  len_data=length(dataOxy2018) % time step is 30 sec
%  data = dataOxy2018(1:numpoints);

XTrain=[X_true(1:end-1)];
YTrain=[X_true(2:end)];

% Define LSTM Network Architecture ------------------------

numFeatures = 1;
numHiddenUnits1 = 16;
numResponses = 7;
numHiddenUnits2 = 100;
numHiddenUnits3 = 120;
numHiddenUnits4 = 100;
numHiddenUnits5 = 120;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits1)
  %  dropoutLayer(0.2)
%     lstmLayer(numHiddenUnits2)
%     dropoutLayer(0.2)
%     lstmLayer(numHiddenUnits3)
%     dropoutLayer(0.2)
%     lstmLayer(numHiddenUnits4)
%     dropoutLayer(0.2)
%     lstmLayer(numHiddenUnits5)
%     dropoutLayer(0.2)
    fullyConnectedLayer(numResponses)];
   % regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',150, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose', 0,...
    'Plots','training-progress');
% Train LSTM Network ---------------------------------------
 %net = trainNetwork(XTrain,YTrain,layers,options);
 %netLSTMBoknis=net;
% 'zdes'
% load netLSTMBoknis
% net=netLSTMBoknis;
%---------------------------------------------------------

%%
dt5=[X_true(1+ind_start:ind_start+numpoints)];
tt5=[t_true(1+ind_start:ind_start+numpoints)];
d_forecast=[X_true(ind_start+numpoints-pred_horizon_t+1:ind_start+numpoints)];
t_forecast=[t_true(ind_start+numpoints-pred_horizon_t+1:ind_start+numpoints)];

t_f=[tt5];
X_f=[dt5];
data5_std=[dt5(2:numpoints)];
data5_train=[dt5(1:numpoints-1)];
rmse_f=[0];
for i=1:numpoints-1
    rmse_f=[rmse_f,0];
end;

dt_begin=dt5;
t5=[t_true(1:numpoints)];
numTimeStepsTest=1;
gr1=0;
gr2=0;
ind_points=1;

for s1=1:numD
  if (t_downscale(s1)>=t_true(1+ind_start) && gr1==0)  
      ind_points_start=s1;
      gr1=1;
    end;
    if (t_downscale(s1)<=t_true(numpoints+ind_start+Nsteps))  
      ind_points_end=s1;
    end;
end  

% hold on
% plot(new_time_sec,X(:,4),'g');
% plot(t_5,X_5,'.');
% plot(t_next,X_next,'s');
% plot(t_next(ind_points), 0,'x');

% figure
% plot(new_time_sec,X(:,4),'g');
% hold on
% plot(t_next,X_next,'.');
% hold on

net = trainNetwork(dt5(1:end-1),dt5(2:end),layers,options);
netLSTMBoknis=net;
[net,YPred] = predictAndUpdateState(net,dt5(end));
 s2=numpoints+ind_start+1;
 t_curr=t_true(s2);
 tt5=[tt5(2:numpoints),t_curr(1)];
dt5=[dt5(2:numpoints),YPred(1)];
d_forecast=[d_forecast(2:pred_horizon_t),YPred(1)];
t_forecast=[t_forecast(2:pred_horizon_t),t_curr(1)];
t_f=[t_f,t_curr(1)];
X_f=[X_f,dt5(numpoints)];
rmse_f=[rmse_f,sqrt((X_true(numpoints+1)-YPred)^2)] ; 
 
 while s2<=numpoints+ind_start+Nsteps
        s2=s2+1;
        t_curr=t_true(s2);
% for j=1:2   

% end;
 for i = 1:1
     net = trainNetwork(dt5(1:end-1),dt5(2:end),layers,options);
     [net,YPred] = predictAndUpdateState(net,dt5(end),'ExecutionEnvironment','cpu');
 end
%  size(t_curr)
%  size(YPred)
%  plot(t_curr,YPred,'o','MarkerEdgeColor',[0.8 0.8 0.4]);
 %net = resetState(net);
 %YPred = sig*YPred + mu;
 %plot(t_curr(1),YPred(1),'s','Markersize',10); 
% t_next
% X_update=[X_update(2:4),X_next(ind_points)];
%YPred=[X_5(s2:s2+3)];
hold on
tt5=[tt5(2:numpoints),t_curr(1)];
t_curr(1)
dt5=[dt5(2:numpoints),YPred(1)];
d_forecast=[d_forecast(2:pred_horizon_t),YPred(1)];
t_forecast=[t_forecast(2:pred_horizon_t),t_curr(1)];
t_f=[t_f,t_curr(1)];
X_f=[X_f,dt5(numpoints)];
rmse_f=[rmse_f,sqrt((dt5(numpoints)-X_true(s2))^2)] ;
for s1=ind_points_start:ind_points_end
   % if (t_next(s1)<=t_5(numpoints+time_start*pred_horizon_t+Nsteps))  
for kk=1:1
if (abs(t_downscale(s1)-t_curr(kk))<=deltaT) 
 
    ind_t=s1;
    ind_points_start=ind_t;
    dt5=[X_true(s2-(numpoints-1):s2)]
    tt5=[t_true(s2-(numpoints-1):s2)];
    
%     data5_train=dt5(1:end-1);
%     data5_std=dt5(2:end);
%    t_f=[t_f,t_curr(1)];
%    X_f=[X_f,dt5(numpoints)];
%    rmse_f=[rmse_f,sqrt((dt5(numpoints)-X_true(s2))^2)] ;
%     % net=resetState(net);
%     % net = trainNetwork(data5_train,data5_std,layers,options);
%     [net,YPred] = predictAndUpdateState(net,dt5(end));
%      s2=s2+1;
%     t_curr=t_true(s2);
%     tt5=[tt5(2:numpoints),t_curr(1)];
%     t_curr(1)
%     dt5=[dt5(2:numpoints),YPred(1)];
%     plot(tt5,dt5,'-s','MarkerSize',10);
%     d_forecast=[d_forecast(2:pred_horizon_t),YPred(1)];
% t_forecast=[t_forecast(2:pred_horizon_t),t_curr(1)];
% t_f=[t_f,t_curr(1)];
% X_f=[X_f,dt5(numpoints)];
% rmse_f=[rmse_f,sqrt((dt5(numpoints)-X_true(s2))^2)] ;
  %  plot(t_curr,YPred,'o','MarkerEdgeColor',[0.8 0.8 0.8],'MarkerSize',10);
    %dt5=[dt5(2:numpoints),X_downscale(s1)]; 
%     for rr=1:numpoints        
%             for rr2=1:numpoints-1
%             if (tt5(rr)>=t_5(rr2+s1*pred_horizon_t))&&(tt5(rr)<t_5(rr2+1+s1*pred_horizon_t))
%                 dt5(rr)=(X_5(rr2+(s1-1)*pred_horizon_t)+X_5(rr2+1+(s1-1)*pred_horizon_t))/2
%              %   t_5(1+time_start*pred_horizon_t:time_start*pred_horizon_t+numpoints)
%         end;
%             end;
%         end;
    end;
end;
%end;
end;

% t_f=[t_f,t_curr(1)];
% X_f=[X_f,dt5(numpoints)];
%plot(t_f,X_f,'o','MarkerEdgeColor',[0.8 0. 0.],'MarkerFaceColor',[0.8 0. 0.]);
%rmse_f=[rmse_f(2:pred_horizon_t-1),sqrt(pow2(abs(YPred-X_5(s2+1)/X_5(s2+1))))];  


% XPred=[XPred,YPred];
% tPred=[tt5,t_curr(1)];

% mu = mean(dt5);
% sig = std(dt5);
%YPred=(YPred-mu)/sig;

%end;
end;

% YTest = dataTest(2:end);
% rmse = sqrt(mean((YPred-YTest).^2));


% figure
% plot(time,data,'b')
% hold on
% plot(t_f,X_f,'.','MarkerEdgeColor','r')