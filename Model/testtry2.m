function [XTest,YTest,net,XTrain,YTrain] = testtry2
filename = "CMAPSSData.zip";
dataFolder = "data";
unzip(filename,dataFolder);

filenamePredictors = fullfile(dataFolder,"train_FD001.txt");
[XTrain,YTrain] = prepareDataTrain(filenamePredictors);

m = min([XTrain{:}],[],2)
M = max([XTrain{:}],[],2)
idxConstant = M == m;

for i = 1:numel(XTrain)
    XTrain{i}(idxConstant,:) = [];
end

mu = mean([XTrain{:}],2);
sig = std([XTrain{:}],0,2);

for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sig;
end

thr = 150;
for i = 1:numel(YTrain)
    YTrain{i}(YTrain{i} > thr) = thr;
end

for i=1:numel(XTrain)
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
XTrain = XTrain(idx);
YTrain = YTrain(idx);

numResponses = size(YTrain{1},1);
featureDimension = size(XTrain{1},1);
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];
maxEpochs = 60;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);
%net = trainNetwork(XTrain,YTrain,layers,options);
load Prednet
filenamePredictors = fullfile(dataFolder,"test_FD001.txt");
filenameResponses = fullfile(dataFolder,"RUL_FD001.txt");
[XTest,YTest] = prepareDataTest(filenamePredictors,filenameResponses);

 for i = 1:numel(XTest)
    XTest{i}(idxConstant,:) = [];
    XTest{i} = (XTest{i} - mu) ./ sig;
    size(XTest{i})
%     YTest{i}(YTest{i} > thr) = thr;
 end
 YPred = predict(net,XTest);%,'MiniBatchSize',1);
%save Prednet net
function [XTest,YTest] = prepareDataTest(filenamePredictors,filenameResponses)

XTest = prepareDataTrain(filenamePredictors);

RULTest = dlmread(filenameResponses);

numObservations = numel(RULTest);

YTest = cell(numObservations,1);
for i = 1:numObservations
    X = XTest{i};
    sequenceLength = size(X,2);
    
    rul = RULTest(i);
    YTest{i} = rul+sequenceLength-1:-1:rul;
end

end
function [XTrain,YTrain] = prepareDataTrain(filenamePredictors)

dataTrain = dlmread(filenamePredictors);

numObservations = max(dataTrain(:,1));

XTrain = cell(numObservations,1);
YTrain = cell(numObservations,1);
for i = 1:numObservations
    idx = dataTrain(:,1) == i;
    
    X = dataTrain(idx,3:end)';
    XTrain{i} = X;
    
    timeSteps = dataTrain(idx,2)';
    Y = fliplr(timeSteps);
    YTrain{i} = Y;
end

end
end