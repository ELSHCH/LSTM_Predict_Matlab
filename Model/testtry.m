dataX=[1,2,4,6,1,7,2,13,11,10,7,6,23,34,2,5,10,9,18,11,23,21;...
       1,2,14,61,11,71,2,10,10,10,10,6,23,34,2,5,10,9,18,11,23,21]';
nVar=size(dataX,2);
lengthT=size(dataX,1);
shiftS=1;
sampleS=5;
i=1;
while sampleS+(i-1)*sampleS-(i-1)*shiftS <= lengthT
    i=i+1;
end;
numWindows= i;
xd=zeros(sampleS,nVar,numWindows);
for i=1:numWindows-1
  for j=1:sampleS  
   xd(j,1:nVar,i)=dataX(j+(i-1)*(sampleS-shiftS),1:nVar);
  end;
end;
xd(1:sampleS,1:nVar,numWindows)=dataX(end-sampleS+1:end,1:nVar);
XTrain={xd(1:sampleS-1,1:nVar,1:numWindows)}
YTrain={xd(2:sampleS,1:nVar,1:numWindows)};
size(YTrain{1},1)
 