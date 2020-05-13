function [t_f,Y_f_mean,Y_f_std]=LSTM_PATTERN_KF_train(time_begin,sampleS,bs,time_end,nVar,t_tr,X_tr,t_down,X_down,numHiddenUnits,number_Epochs,ind_predictors,ind_responses,net)
%% Find start and end time indices for filtered time series, 
%  The interval t_filter(ind_point_start:ind_point_end) should overlap t_true(time_start)
 numEpochs=number_Epochs;
 numFeatures=length(ind_predictors);
 numResponses=length(ind_responses);
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
    'MaxEpochs',numEpochs, ...
    'GradientThreshold',1, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Verbose',0);% ...
Y_f=zeros(time_end-time_begin+1,bs,numResponses);
Y_f_mean=zeros(time_end-time_begin+1,numResponses);
Y_f_std=zeros(time_end-time_begin+1,numResponses);

lengthWindow=time_end-time_begin+1;
%[XTrain,YTrain,net] = prepareTrainData(X_down,t_down,sampleS);

t_f=t_down(time_begin:time_end);

i=1;
%shiftS=sampleS-1; % overlap between running windows of size sampleS 
 shiftS = 0; % no overlap option
lengthT=floor(length(t_down)); % length of time series

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
hold on
for i=1:numWindows-1
  for j=1:sampleS   
   xd(j,1:numFeatures,i)=X_down(j+i*sampleS-i*shiftS,ind_predictors);
   td(j,i)=t_down(j+(i-1)*sampleS-(i-1)*shiftS);
   xd_explain(j,1:numFeatures,i)=X_down(j+(i-1)*sampleS-(i-1)*shiftS,ind_predictors);
  end;
end;
hold on
for j=1:numWindows-1
  [C,id]=intersect(td(:,j),t_down(time_begin+1));
  if isnan(id)==0
     idWindow=j;
  end;
 % plot(td(:,j),xd(:,1,j));
end;

F=eye(numResponses);
H= eye(numResponses);
P = eye(numResponses)* 100; % initialize uncertainty of covariance
Q_init= eye(numResponses)* 100;
hold on
for i=1:numWindows-1
    
    for i2=1:sampleS 
     for i1=1:numFeatures
        xq_explain(i1,i2)=xd_explain(i2,i1,i); % add uncertainty for flexible predicton
     end;
     for i1=1:numFeatures
        xq(i1,i2)=xd(i2,i1,i); % add uncertainty for flexible predicton
     end;
    end;  
    
    XTest={[xq_explain(1:numFeatures,1:lengthWindow)]}; % explained parameters 
    for tr=1:1 
     net=resetState(net);   
 
     YPred = predict(net,XTest,'MiniBatchSize',1); % estimate responses/predictions
     
     X  = cell2mat(YPred); % convert from cell to array format   
     
     if tr==1
      R_init = sampleCovariance(X',numResponses,lengthWindow); % covariance of time series 
  
      Q_std = Q_init; % covariance of state
      R_std = R_init; % covariance of time series 
   
      P = F*P*transpose(F)+Q_std; % initialize uncertainty of covariance
    end;
    
    for ii=1:numResponses  
     for j=1:lengthWindow
      z(ii,j)=X(ii,j)+R_std(ii,ii);
     end;
    end;
   
    y = z - H*X;
  
 % Estimate Kalman Gain
     K = P*transpose(H)*inv(H*P*transpose(H)+R_std);
     
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

     P = F*P*transpose(F)+eye(numResponses).*Q_std; % initialize uncertainty of covariance (first guess)
   
     for i3=1:numResponses  
       for j=1:lengthWindow
         z(i3,j)=X(i3,j)+R_std(i3,i3);
       end;
     end;
    
     y = z - H*X;
    
 % Estimate Kalman Gain
     K = P*transpose(H)*inv(H*P*transpose(H)+eye(numResponses).*R_std);
 % predict new state with the Kalman Gain correction

     XPred = X + K*y;%+xd_explain(1:lengthWindow,1:numFeatures,i)';  
    
     P = (eye(numResponses) - K*H)*P*transpose((eye(numResponses) - K*H)) + K*eye(numResponses).*R_std*transpose(K); % uncertainty of covariance correction of first guess

     for i4=1:numResponses
       p_diag(1:lengthWindow,i4,i)=P(i4,i4);
     end;
        
    end;
   % plot(td(1:lengthWindow,i),XPred(1:numResponses,:),'g');
    Y_f(:,1,1:numResponses)=XPred(1:numResponses,:)';
if i==idWindow+1     
for i1=1:numResponses
  for i2=1:lengthWindow  
     Y_f_mean(i2,i1)=Y_f(i2,1,i1);
     Y_f_std(i2,i1)=Y_f(i2,1,i1);
  end;
end;  
end;
end;
t_f=td(1:lengthWindow,idWindow+1);
