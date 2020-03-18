function [t_f,Y_f_mean,Y_f_std]=LSTM_PATTERN(time_begin,sampleS,bs,time_end,nVar,t_tr,X_tr,t_down,X_down,numFeatures,numResponses,numHiddenUnits,number_Epochs,net)
%% Find start and end time indices for filtered time series, 
%  The interval t_filter(ind_point_start:ind_point_end) should overlap t_true(time_start)
number_Epochs=numEpochs;
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
i=1;
%shiftS=sampleS-1; % overlap between running windows of size sampleS 
 shiftS = 0; % no overlap option
lengthT=floor(length(t_down)); % length of time series
t_f=t_down(time_begin:time_end);
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
   xd(j,1:numFeatures,i)=X_down(j+i*sampleS-i*shiftS,1:numFeatures);
   td(j,i)=t_down(j+(i-1)*sampleS-(i-1)*shiftS);
   xd_explain(j,1:numFeatures,i)=X_down(j+(i-1)*sampleS-(i-1)*shiftS,1:numFeatures);
  end;
end;
% Find index of time window in which  t_down(time_begin+1) is located  
for j=1:numWindows-1
  [C,id]=intersect(td(:,j),t_down(time_begin+1));
  if isnan(id)==0
     idWindow=j;
  end;   
end;
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
     net=resetState(net); 
     size(xq_explain(1:numFeatures,1:lengthWindow))
     YPred = predict(net,XTest,'MiniBatchSize',1); % estimate responses/predictions
     Y_ff(:,1,1:numResponses)=cell2mat(YPred(1:numResponses,:)');%+xq_explain(1:numResponses,lengthWindow)';
     plot(td(1:lengthWindow,i),Y_ff(:,1,1),'y'); 
   if i==idWindow      
     Y_f(:,1,1:numResponses)=cell2mat(YPred(1:numResponses,:)');
     %  plot(td(1:lengthWindow,n_r),Y_f(:,b_r,1),'y'); 
   end;
end;   
for i=1:lengthWindow
 for j=1:numResponses
      Y_f_mean(i,j)=mean(Y_f(i,:,j));
      Y_f_std(i,j)=mean(Y_f(i,:,j));    
    %  plot(td(:,idWindow),0,'x');
 end;  
end;
t_f=td(1:lengthWindow,idWindow+1);