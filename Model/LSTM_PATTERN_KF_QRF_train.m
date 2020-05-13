function [net_KF,net_F]=LSTM_PATTERN_KF_QRF_train(time_begin,sampleS,bs,time_end,P_horizon_s,t_down,X_down,...
                                        numHiddenUnits,number_Epochs,ind_predictors,ind_responses,net)
                                   
%------------------------------------------------------------------------------------
%   LSTM model combined with Kalman Filter corrections (LSTM trains for covariance of predictions 
%                                                       and for covariance of time series) 
%------------------------------------------------------------------------------------
%  Input parameters: nVar - number of time series
%                    sampleS - length of prediction horizon (in sec)
%                    net - pretrained network using entire time series
%                          X_down
%
% Last modified 24.02.2020 E.Shchekinova
%-------------------------------------------------------------------------------------
% Define options for training a small LSTM network model only for time series (size nVar) 
 numEpochs=number_Epochs;
 numResponses=length(ind_responses);
 numFeatures=length(ind_predictors);
 layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
%     dropoutLayer(0.7)
%     lstmLayer(numHiddenUnits)
%     dropoutLayer(0.7)
%     lstmLayer(numHiddenUnits)
    dropoutLayer(0.7)
    fullyConnectedLayer(numResponses)
   % fullyConnectedLayer(numFeatures)
    regressionLayer];
 miniBatchSize = 20;
 options = trainingOptions('sgdm', ...
    'MaxEpochs',numEpochs, ...
    'GradientThreshold',1, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0);% ...
%load NetworkQRFOxyWindWindDirPress48
%load NetworkQRFOxy48_4Var
%load NetworkQRFOxy48_3_1Var
% Initialize prediction time series of a given sizes 
lengthWindow=time_end-time_begin+1; % length of prediction horizion ( the same as sampleS here)

Y_f=zeros(lengthWindow,bs,numResponses);
Y_f_mean=zeros(lengthWindow,numResponses);
Y_f_std=zeros(lengthWindow,numResponses);
% Define prediction time
t_f=t_down(time_begin:time_end);

%[XTrain,YTrain,net] = prepareTrainData(X_down,t_down,nVar,nVar,sampleS);
i=1;
%shiftS=sampleS-1; % overlap between running windows of size sampleS 
 shiftS = 0; % no overlap option
lengthT=floor(length(X_down)); % length of time series

hold on
% Estimate number of time windows  (size sampleS) for time series of length - lengthT  
while sampleS+(i-1)*sampleS-(i-1)*shiftS <= lengthT
    i=i+1;
end;

numWindows=i-1; % number of time windows 
% 
xd=zeros(sampleS,numFeatures,numWindows-1);
td=zeros(sampleS,numWindows-1); % time windows 
xd_explain=zeros(sampleS,numFeatures,numWindows-1);
xq=zeros(numFeatures,sampleS);
xq_explain=zeros(numFeatures,sampleS);
% save time series into fixed time windows of size (sampleS X nVar X numWindows)
%bW=floor(numWindows/2);
for i=1:numWindows-1
   xd_explain(1:sampleS,1:numFeatures,numWindows-i)=X_down(1+lengthT-((i+1)*sampleS-(i+1)*shiftS):lengthT-(i*sampleS-i*shiftS),ind_predictors);
   xd(1:sampleS,1:numFeatures,numWindows-i)=X_down(1+lengthT-(i*sampleS-i*shiftS):lengthT-((i-1)*sampleS-(i-1)*shiftS),ind_predictors);
   td(:,numWindows-i)=t_down(1+lengthT-((i+1)*sampleS-(i+1)*shiftS):lengthT-(i*sampleS-i*shiftS));
 % plot(td(1:sampleS,numWindows-i),xd(1:sampleS,1,numWindows-i)+1,'x');
end;
%-----------Define array td of time windows of fixed length lengthWindow to
%           cover entire time series t_down(1:lengthT) and forward prediction interval
%           of lengthWindow
%-------Definition of two last intervals td(1:lengthWindow,numWindows:numWindows+1) corresponding to prediction horizon 
delt_td=td(2,1)-td(1,1); % time step
for i=1:sampleS
 td(i,numWindows)=td(sampleS,numWindows-1)+i*delt_td;
end;
for i=1:sampleS
 td(i,numWindows+1)=td(sampleS,numWindows)+i*delt_td;
end;
%----------------------------------------------------------------------------------------------------------
% hold on
% for i=1:numWindows-1
%     plot(td(1:sampleS,i+1),xd(1:sampleS,1,i)+1,'x');
%     plot(td(1:sampleS,i),xd_explain(1:sampleS,1,i),'o');
% end;    
% Initialize several arrays where prediction data will be saved
Q_run=[];
R_run=[];
P_run=[];
X_run=[];
X_exp=[];
X_res=[];

F=eye(numFeatures); % unitary matrix 
H= eye(numFeatures); % unitary matrix
P = eye(numFeatures)*100; % initialize uncertainty of covariance , initially 100%

% Estimate prediction covariance and covariance of data averaged for every time window of size sampleS
hold on;
for i=1:numWindows-1

    for i2=1:sampleS 
     for i1=1:numFeatures
        xq_explain(i1,i2)=xd_explain(i2,i1,i); % add uncertainty for flexible predicton
     end;
     for i1=1:numResponses
        xq(i1,i2)=xd(i2,i1,i); % add uncertainty for flexible predicton
     end;
    end;  
    XTest={[xq_explain(1:numFeatures,1:lengthWindow)]}; % explained parameters 
%     if i==1
%     plot(td(1:lengthWindow,i),xq_explain(1,1:lengthWindow),'o');
%     plot(td(1:lengthWindow,i+1),xq(1,1:lengthWindow),'x');
%     end; 

    for tr=1:1 
     net=resetState(net);   

     YPred = predict(net,XTest,'MiniBatchSize',1); % estimate responses/predictions
   
     X  = cell2mat(YPred); % convert from cell to array format   
     Q_std = covarianceStateTransitionPattern(xq(1:numFeatures,1:lengthWindow),...
            X,numFeatures,lengthWindow); % covariance of state
%         if tr==1
%         plot(td(1:lengthWindow,i),xq(1,1:lengthWindow),'b');
%         plot(td(1:lengthWindow,i),X(1,1:lengthWindow),'b');
%         end;
        
     R_std = sampleCovariance(xq',numFeatures,lengthWindow); % covariance of time series 
     for i2=1:numFeatures
         R_d(1:lengthWindow,i2,i)=R_std(i2,i2);
         Q_d(1:lengthWindow,i2,i)=Q_std(i2,i2);
     end;
 
     P = F*P*transpose(F)+eye(numFeatures).*Q_std; % initialize uncertainty of covariance (first guess)
   
     for i3=1:numFeatures  
       for j=1:lengthWindow
         z(i3,j)=X(i3,j)+R_std(i3,i3);
       end;
     end;
    
     y = z - H*X;
 
 % Estimate Kalman Gain
     K = P*transpose(H)*inv(H*P*transpose(H)+eye(numFeatures).*R_std);
 % predict new state with the Kalman Gain correction

     XPred = X + K*y;%+xd_explain(1:lengthWindow,1:numFeatures,i)';  
    
     P = (eye(numFeatures) - K*H)*P*transpose((eye(numFeatures) - K*H)) + K*eye(numFeatures).*R_std*transpose(K); % uncertainty of covariance correction of first guess

     for i4=1:numFeatures
       p_diag(1:lengthWindow,i4,i)=P(i4,i4);
     end;
        
    % plot(td(1:lengthWindow,i),XPred(1,1:lengthWindow),'g');

    end;
    % Save covariances of prediction, time series and uncertaity of
    % covariance for every fixed window (in total numWindows)
     Q_run=[Q_run';Q_d(1:lengthWindow,1:numFeatures,i)]'; % 
     R_run=[R_run';R_d(1:lengthWindow,1:numFeatures,i)]'; %
     P_run=[P_run';p_diag(1:lengthWindow,1:numFeatures,i)]'; % 
end;
% Use estimates obtained in previos loop to define joint LSTM model for time
% series and covariances, estimates Q_run, R_run will be used for model training 

% Find index of time window in which t_down(time_begin) is located  
for j=1:numWindows+1
  if (~isempty(find(td(1,j)<t_down(time_begin)+P_horizon_s)))&&(~isempty(find(td(sampleS,j)>t_down(time_begin)+P_horizon_s)))
    idWindow=j;
  end;  
end; 

clear Q_explain R_explain Q_r R_r xq_explain xq 
Q_explain=eye(numFeatures);
R_explain=eye(numFeatures);
Q_r=eye(numResponses);
R_r=eye(numResponses);

% for i=1:numResponses
%  Q_r(i,i)=Q_d(1,i,idWindow);
%  R_r(i,i)=R_d(1,i,idWindow);
%  Q_r_d(i)=Q_d(1,i,idWindow);
%  R_r_d(i)=R_d(1,i,idWindow);
% end; 
% for i=1:numFeatures
%  Q_explain(i,i)=Q_d(1,i,idWindow-1);
%  R_explain(i,i)=R_d(1,i,idWindow-1);
%  Q_explain_d(i)=Q_d(1,i,idWindow-1);
%  R_explain_d(i)=R_d(1,i,idWindow-1);
% end; 
hold on;
for i=1:numWindows-2
 for s1=1:numResponses
  Q_r(s1,s1)=Q_d(1,s1,i+1);
  R_r(s1,s1)=R_d(1,s1,i+1);
  Q_r_d(s1)=Q_d(1,s1,i+1);
  R_r_d(s1)=R_d(1,s1,i+1);
 end; 
 for s1=1:numFeatures
  Q_explain(s1,s1)=Q_d(1,s1,i);
  R_explain(s1,s1)=R_d(1,s1,i);
  Q_explain_d(s1)=Q_d(1,s1,i);
  R_explain_d(s1)=R_d(1,s1,i);
 end;        
    for i2=1:sampleS 
     for i1=1:numFeatures
        xq_explain(i1,i2)=xd_explain(i2,i1,i);
     end;
    for i1=1:numResponses
   % for i1=1:numFeatures
        xq(i1,i2)=xd(i2,i1,i); 
     end;
    end;  

%     X_exp=[xq_explain(1:nVar,1:lengthWindow);Q_d(1:lengthWindow,1:nVar,i)';R_d(1:lengthWindow,1:nVar,i)'];
%     X_res=[xq;Q_d(1:lengthWindow,1:nVar,i+1)';R_d(1:lengthWindow,1:nVar,i+1)'];
   if i==1
    plot(td(1:lengthWindow,i),xq_explain(1,1:lengthWindow),'o');
    plot(td(1:lengthWindow,i+1),xq(1,1:lengthWindow),'x');
   end; 

    X_exp=[xq_explain(1:numFeatures,1:lengthWindow)';Q_explain_d(1:numFeatures);...
           R_explain_d(1:numFeatures)]';
    X_res=[xq';Q_r_d(1:numResponses);R_r_d(1:numResponses)]';
%     X_exp=[xq_explain(1:numFeatures,1:lengthWindow)']';
%     X_res=[xq']';
    XTrain_1(i)={X_exp}; 
    YTrain_1(i)={X_res};
   
%     plot(td(1:lengthWindow,i+1),xq(1,1:lengthWindow),'k');
%     plot(td(1:lengthWindow,i),xq_explain(1,1:lengthWindow),'r');
%     if i==idWindow
%       plot(td(1:lengthWindow,i),xq_explain(1,1:lengthWindow),'y');
%     end;
  
end; 
 %clear XTrain YTrain
 
 XTrain=XTrain_1; % define explained parameters array 
 YTrain=YTrain_1; % define responses array
 
% Set network options with 3*nVar number of inputs and outputs ( nVar time series, nVar covariances of prediction, nVar covariances of time series  
 layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(100,'OutputMode','sequence')
    dropoutLayer(0.7)
    lstmLayer(numHiddenUnits)
    dropoutLayer(0.7)
    lstmLayer(numHiddenUnits)
    dropoutLayer(0.7)
    fullyConnectedLayer(numResponses)
  %  fullyConnectedLayer(numFeatures)
    regressionLayer];
 miniBatchSize = 100;
 options = trainingOptions('sgdm', ...
    'MaxEpochs',numEpochs, ...
    'GradientThreshold',0.11, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0);% ...

% Train LSTM network with input size 3*nVar 
netQRF = trainNetwork(XTrain,YTrain,layers,options);
% resetState(netQRF);
% X=predict(netQRF,XTrain);
% save tr X XTrain YTrain
net_F=net;
net_KF=netQRF;