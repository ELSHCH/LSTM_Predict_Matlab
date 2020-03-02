function [t_f,Y_f_mean,Y_f_std]=LSTM_PATTERN_KF_QRF(time_begin,sampleS,bs,time_end,nVar,t_tr,X_tr,t_down,X_down,net)
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
'zdes'
 numResponses = nVar;
 numFeatures = nVar;
 numHiddenUnits = 100;
 layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
%     dropoutLayer(0.7)
%     lstmLayer(numHiddenUnits)
%     dropoutLayer(0.7)
%     lstmLayer(numHiddenUnits)
    dropoutLayer(0.7)
    fullyConnectedLayer(numResponses)
    regressionLayer];
 miniBatchSize = 20;
 options = trainingOptions('sgdm', ...
    'MaxEpochs',100, ...
    'GradientThreshold',1, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Verbose',0);% ...
%load NetworkQRF48 
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
lengthT=length(t_down/2); % length of time series

% Estimate number of time windows  (size sampleS) for time series of length - lengthT  
while sampleS+(i-1)*sampleS-(i-1)*shiftS <= lengthT
    i=i+1;
end;
numWindows=i-1; % number of time windows 

% 
xd=zeros(sampleS,nVar,numWindows-1);
td=zeros(sampleS,numWindows-1); % time windows 
xd_explain=zeros(sampleS,nVar,numWindows-1);
xq=zeros(nVar,sampleS);
xq_explain=zeros(nVar,sampleS);
% save time series into fixed time windows of size (sampleS X nVar X numWindows)

%bW=floor(numWindows/2);
hold on
for i=1:numWindows-1
  for j=1:sampleS   
   xd(j,1:numResponses,i)=X_down(j+i*sampleS-i*shiftS,1:nVar);
   td(j,i)=t_down(j+(i-1)*sampleS-(i-1)*shiftS);
   xd_explain(j,1:nVar,i)=X_down(j+(i-1)*sampleS-(i-1)*shiftS,1:nVar);
  end;
end;
% Initialize several arrays where prediction data will be saved
Q_run=[];
R_run=[];
P_run=[];
X_run=[];
X_exp=[];
X_res=[];
F=eye(nVar); % unitary matrix 
H= eye(nVar); % unitary matrix
P = eye(nVar)*100; % initialize uncertainty of covariance , initially 100%

% Iteration loops over numWindows and estimates prediction covariance and
% covariance of data averaged for every time window of size sampleS
hold on;

for i=1:numWindows-1
    numWindows
    for i2=1:sampleS 
     for i1=1:nVar
        xq_explain(i1,i2)=xd_explain(i2,i1,i); % add uncertainty for flexible predicton
     end;
     for i1=1:numResponses
        xq(i1,i2)=xd(i2,i1,i); % add uncertainty for flexible predicton
     end;
    end;  
    XTest={[xq_explain(1:nVar,1:lengthWindow)]}; % explained parameters 
    for tr=1:1 
     net=resetState(net);   
     YPred = predict(net,XTest,'MiniBatchSize',1); % estimate responses/predictions

     X  = cell2mat(YPred); % convert from cell to array format   
     Q_std = covarianceStateTransitionPattern(xq(1:numResponses,1:lengthWindow),...
            X,numResponses,lengthWindow); % covariance of state
%         if tr==1
%         plot(td(1:lengthWindow,i),xq(1,1:lengthWindow),'b');
%         plot(td(1:lengthWindow,i),X(1,1:lengthWindow),'b');
%         end;
        
     R_std = sampleCovariance(xq',numResponses,lengthWindow); % covariance of time series 
     for i2=1:numResponses
         R_d(1:lengthWindow,i2,i)=R_std(i2,i2);
         Q_d(1:lengthWindow,i2,i)=Q_std(i2,i2);
     end;

     P = F*P*transpose(F)+eye(nVar).*Q_std; % initialize uncertainty of covariance (first guess)
   
     for i3=1:nVar  
       for j=1:lengthWindow
         z(i3,j)=X(i3,j)+R_std(i3,i3);
       end;
     end;
    
     y = z - H*X;

 % Estimate Kalman Gain
     K = P*transpose(H)*inv(H*P*transpose(H)+eye(nVar).*R_std);
 % predict new state with the Kalman Gain correction

     XPred = X + K*y+xd_explain(1:lengthWindow,1:nVar,i)';  
    
     P = (eye(nVar) - K*H)*P*transpose((eye(nVar) - K*H)) + K*eye(nVar).*R_std*transpose(K); % uncertainty of covariance correction of first guess

     for i4=1:nVar
       p_diag(1:lengthWindow,i4,i)=P(i4,i4);
     end;
        
     plot(td(1:lengthWindow,i),XPred(1,1:lengthWindow),'g');

    end;
    % Save covariances of prediction, time series and uncertaity of
    % covariance for every fixed window (in total numWindows)
     Q_run=[Q_run';Q_d(1:lengthWindow,1:nVar,i)]'; % 
     R_run=[R_run';R_d(1:lengthWindow,1:nVar,i)]'; %
     P_run=[P_run';p_diag(1:lengthWindow,1:nVar,i)]'; % 
end;
% Use estimates obtained in previos loop to define joint LSTM model for time
% series and covariances, estimates Q_run, R_run will be used for model training 

% Find index of time window in which  t_down(time_begin+1) is located  
for j=1:numWindows-1
  [C,id]=intersect(td(:,j),t_down(time_begin+1));
  if isnan(id)==0
     idWindow=j;
  end;   
end;
Q_explain=eye(nVar);
R_explain=eye(nVar);
Q_r=eye(nVar);
R_r=eye(nVar);

for i=1:nVar
 Q_r(i,i)=Q_d(1,i,idWindow);
 R_r(i,i)=R_d(1,i,idWindow);
 Q_explain(i,i)=Q_d(1,i,idWindow-1);
 R_explain(i,i)=Q_d(1,i,idWindow-1);
end; 
hold on;
for i=1:numWindows-2
    for i2=1:sampleS 
     for i1=1:nVar
        xq_explain(i1,i2)=xd_explain(i2,i1,i)*(1+0*rand(1));
     end;
     for i1=1:numResponses
        xq(i1,i2)=xd(i2,i1,i)*(1+0*rand(1)); 
     end;
    end;  
    
%     X_exp=[xq_explain(1:nVar,1:lengthWindow);Q_d(1:lengthWindow,1:nVar,i)';R_d(1:lengthWindow,1:nVar,i)'];
%     X_res=[xq;Q_d(1:lengthWindow,1:nVar,i+1)';R_d(1:lengthWindow,1:nVar,i+1)'];
    X_exp=[xq_explain(1:nVar,1:lengthWindow)';Q_explain(1:nVar,1:nVar);R_explain(1:nVar,1:nVar)]';
    X_res=[xq';Q_r(1:nVar,1:nVar);R_r(1:nVar,1:nVar)]';
    XTrain_1(i)={X_exp}; 
    YTrain_1(i)={X_res};

%     plot(td(1:lengthWindow,i+1),xq(1,1:lengthWindow),'k');
%     plot(td(1:lengthWindow,i),xq_explain(1,1:lengthWindow),'r');
%     if i==idWindow
%       plot(td(1:lengthWindow,i),xq_explain(1,1:lengthWindow),'y');
%     end;
  
end; 

 XTrain=XTrain_1; % define explained parameters array 
 YTrain=YTrain_1; % define responses array
 
% Set network options with 3*nVar number of inputs and outputs ( nVar time series, nVar covariances of prediction, nVar covariances of time series  
 numResponses = nVar;
 numFeatures = nVar;
 numHiddenUnits = 100;
 layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
%     dropoutLayer(0.7)
%     lstmLayer(numHiddenUnits)
%     dropoutLayer(0.7)
%     lstmLayer(numHiddenUnits)
    dropoutLayer(0.7)
    fullyConnectedLayer(numResponses)
    regressionLayer];
 miniBatchSize = 20;
 options = trainingOptions('sgdm', ...
    'MaxEpochs',100, ...
    'GradientThreshold',1, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Verbose',0);% ...

% Train LSTM network with input size 3*nVar 
 netQRF = trainNetwork(XTrain,YTrain,layers,options);
  
% Initialize unitary matrices  
  P_init=eye(nVar)*100;
  Q_std = eye(nVar)*100;
  R_std = eye(nVar)*100;

for i=1:nVar
  P_init(i,i)=0;
  Q_std(i,i) = 0;
  R_std(i,i) = 0;
end;    
% for i=1:nVar
%   P_init(i,i)=P_run(i,(idWindow)*sampleS); 
% end;
for n_r=1:numWindows-2
    n_r,numWindows
for b_r=1:bs   
   b_r
%    X_iter=[(1+rand(1))*X_down(time_begin-lengthWindow+1:time_begin,1:nVar)';Q_run(1:nVar,(idWindow)*sampleS+1:(idWindow+1)*sampleS);...
%           R_run(1:nVar,(idWindow)*sampleS+1:(idWindow+1)*sampleS)]; 
 %   X_iter=[(1+rand(1))*X_down(time_begin-lengthWindow+1:time_begin,1:nVar);Q_explain(1:nVar,1:nVar);...
  %         R_explain(1:nVar,1:nVar)]; 
     X_iter=[xd(1:lengthWindow,1:nVar,n_r);Q_d(1:lengthWindow,1:nVar,n_r);R_d(1:lengthWindow,1:nVar,n_r)];
   for i=1:lengthWindow
    for k=1:nVar   
     X_d(time_begin-lengthWindow+1:time_begin,k)=X_down(time_begin-lengthWindow+1:time_begin,k); 
    end; 
   end;    
  
    XTest=X_iter';
 for tr =1:2      
    netQRF=resetState(netQRF);
    YPred = predict(netQRF,XTest,'MiniBatchSize',1);
    X  = YPred; % estimate new state using network prediction

    
    for i1=1:nVar 
      Q_std(i1,i1) = X(i1,lengthWindow+i1); % covariance of state
      R_std(i1,i1) = X(i1,lengthWindow+nVar+i1); % covariance of time series 
     end; 
    Q_std=eye(nVar).*Q_std; 
    R_std=eye(nVar).*R_std; 
    P = F*P_init*transpose(F)+Q_std; % initialize uncertainty of covariance
   
    for i=1:nVar  
     for j=1:lengthWindow
      z(i,j)=X(i,j)+R_std(i,i);
     end;
    end;
    y = z - H*X(1:nVar,1:lengthWindow);
  
 % Estimate Kalman Gain
    K = P*transpose(H)*inv(H*P*transpose(H)+R_std);
 % predict new state with the Kalman Gain correction

    XPred = X(1:nVar,1:lengthWindow) + K*y+xd_explain(1:lengthWindow,1:nVar,n_r)';  
   
    P = (eye(nVar) - K*H)*P*transpose((eye(nVar) - K*H)) + K*R_std*transpose(K);

% X_exp=[xq_explain(1:nVar,1:lengthWindow)';Q_std ;R_std]';
Q_std = covarianceStateTransitionPattern(xd(1:lengthWindow,1:nVar,n_r)',...
         XPred(1:nVar,1:lengthWindow),numResponses,lengthWindow);
for kk=1:nVar
 %Q_dd(kk,kk)=Q_d(1,kk,n_r); 
 R_dd(kk,kk)=R_d(1,kk,n_r);
% R_std = sampleCovariance(xq',numResponses,lengthWindow); % covariance of time series 
end;
 X_exp=[xd(1:lengthWindow,1:nVar,n_r);Q_std;R_dd]';
 X_res=[xd(1:lengthWindow,1:nVar,n_r+1);X(1:nVar,lengthWindow+1:lengthWindow+nVar);X(1:nVar,lengthWindow+nVar+1:lengthWindow+2*nVar)]';
  for s=1:numWindows
    XTrain_1(i)={X_exp}; 
    YTrain_1(i)={X_res};
  end; 
  XTrain=XTrain_1'; % define explained parameters array 
  YTrain=YTrain_1'; % define responses array
% netQRF = trainNetwork(XTrain,YTrain,layers,options);
%   X_iter=[X_down(time_begin-lengthWindow+1:time_begin,1:nVar);X(1:nVar,lengthWindow+1:lengthWindow+nVar);...
%           X(1:nVar,lengthWindow+nVar+1:lengthWindow+2*nVar)]; 
  Q_std= X(1:nVar,lengthWindow+1+nVar:lengthWindow+2*nVar);
  R_std= X(1:nVar,lengthWindow+2*nVar+1:lengthWindow+3*nVar);
  Q_std=Q_std.*eye(nVar);
  R_std=R_std.*eye(nVar);
  Y_f(:,b_r,1:nVar)=XPred(1:nVar,:)';
%     if tr==1   
%      plot(td(1:lengthWindow,n_r),Y_f(:,b_r,1),'y');
% %     elseif tr==2 
% %         plot(td(1:lengthWindow,n_r),Y_f(:,b_r,1),'r');
% %     elseif tr==3 
% %         plot(td(1:lengthWindow,n_r),Y_f(:,b_r,1),'b');    
% %     elseif tr==4 
% %         plot(td(1:lengthWindow,n_r),Y_f(:,b_r,1),'m');  
%      end;    
end;   

   Y_f(:,b_r,1:nVar)=XPred(1:nVar,:)';
   if n_r==idWindow
      Y_f_mean(1:lengthWindow,1:nVar)=Y_f(:,b_r,1:nVar);
      Y_f_std(1:lengthWindow,1:nVar)=Y_f(:,b_r,1:nVar);    
      hold on
    %  plot(td(:,idWindow),0,'x');
   end;   
       plot(td(1:lengthWindow,n_r),Y_f(:,b_r,1),'y');
end; 
end;

% for i1=1:nVar
%   for i2=1:lengthWindow  
%      Y_f_mean(i2,i1)=mean(Y_f(i2,:,i1));
%      Y_f_std(i2,i1)=std(Y_f(i2,:,i1));
%   end;
% end; 
t_f=td(1:lengthWindow,idWindow);
save NetworkQRF3 netQRF net