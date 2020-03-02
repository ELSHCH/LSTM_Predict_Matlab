function net=trainNetworkQRF(X_explain,X_response)
%------------------------------------------------------------------------------ 
%  Prepare set of data for training from time sequence and train network         
%------------------------------------------------------------------------------
%   Input variables: dataX, datatT - original data and time series 
%                    sampleS       - size of fixed window for training sample;                     
%
%   Output variables: net - trained network
%   Last modified E. Shchekinova 18.02.2020
%------------------------------------------------------------------------------%
nVar=size(X_explain,2);
lengthT=size(X_explain,1);
shiftS=0;
i=1;
numResponses = 3;
numFeatures = nVar;
numHiddenUnits = 10;

while sampleS+(i-1)*sampleS-(i-1)*shiftS <= lengthT
    i=i+1;
end;
numWindows=i-1;

xd=zeros(sampleS,numResponses,numWindows-1);
xd_explain=zeros(sampleS,nVar,numWindows-1);
xq=zeros(numResponses,sampleS);
xq_explain=zeros(nVar,sampleS);
for i=1:numWindows-1
  for j=1:sampleS   
   xd(j,1:numResponses,i)=dataX(j+i*sampleS,1:numResponses);
   xd_explain(j,1:nVar,i)=dataX(j+(i-1)*sampleS,1:nVar);
  end;
end;
%  xd(1,1:sampleS,numWindows)=dataX(end-sampleS+1:end,1);
%  for k1=1:nVar
%   for k2=1:sampleS   
%     xd_explain(k1,numWindows,k2)=dataX(end-sampleS+k2,k1+1);
%   end;
%  end;
 for i=1:numWindows-1
    for i2=1:sampleS 
     for i1=1:nVar
        xq_explain(i1,i2)=xd_explain(i2,i1,i)*(1+rand(1));
     end;
     for i1=1:numResponses
        xq(i1,i2)=xd(i2,i1,i)*(1+rand(1)); 
     end;
    end;  
  XTrain_1(i)={xq_explain(1:nVar,1:sampleS)};
  YTrain_1(i)={xq(1:numResponses,1:sampleS)};
  
 end;
 XTrain=XTrain_1';
 YTrain=YTrain_1';

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
    'MaxEpochs',10, ...
    'GradientThreshold',1, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Verbose',0);% ...

net = trainNetwork(XTrain,YTrain,layers,options);