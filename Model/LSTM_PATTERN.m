function [t_f,Y_f_mean,Y_f_std]=LSTM_PATTERN(time_begin,sampleS,bs,time_end,nVar,t_tr,X_tr,t_down,X_down,net)
%% Find start and end time indices for filtered time series, 
%  The interval t_filter(ind_point_start:ind_point_end) should overlap t_true(time_start)

 numResponses = 3;
 numFeatures = nVar;
 numHiddenUnits = 200;
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
Y_f=zeros(time_end-time_begin+1,bs,numResponses);
Y_f_mean=zeros(time_end-time_begin+1,numResponses);
Y_f_std=zeros(time_end-time_begin+1,numResponses);

lengthWindow=time_end-time_begin+1;
%[XTrain,YTrain,net] = prepareTrainData(X_down,t_down,sampleS);

t_f=t_down(time_begin:time_end);
for b_r=1:bs   
   b_r
   for i=1:lengthWindow
    for k=1:nVar   
     X_d(time_begin-lengthWindow+1:time_begin,k)=...
         (1+2*rand(1))*X_down(time_begin-lengthWindow+1:time_begin,k); 
    end; 
   end;    
   XTest={[X_d(time_begin-lengthWindow+1:time_begin,1:nVar)']};
 
   YPred = predict(net,XTest,'MiniBatchSize',1);
   
   X_f=cell2mat(YPred)';

   Y_f(:,b_r,1:numResponses)=X_f(:,1:numResponses);
   
    plot(t_f(1:end),Y_f(:,b_r,1),'g');
    hold on
end; 
for i1=1:numResponses
  for i2=1:lengthWindow  
     Y_f_mean(i2,i1)=mean(Y_f(i2,:,i1));
     Y_f_std(i2,i1)=std(Y_f(i2,:,i1));
  end;
end;  