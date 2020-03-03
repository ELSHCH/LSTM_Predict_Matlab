cd('C:/Users/eshchekinova/Documents/BoknisData/LSTMPred/PredictionData');
%load('NormalizedBoknis.mat');
load('NormalizedBoknis26-Sep-2019.mat');
for i=1:length(new_time)
m_r=datestr(new_time(i));
dm(i)=cellstr(m_r(1:11));
end;
lengthT=floor(length(new_time)/5);
for i=1:3
 ax(i)=subplot(3,1,i);
 plot(ax(i),new_time_sec,X(:,i));
 hold on
 plot(ax(i),new_time_sec(4*lengthT+1:lengthT*5),X(4*lengthT+1:lengthT*5,i),'r');
xticks(new_time_sec(4*lengthT+1:100:end));
xticklabels(dm(4*lengthT+1:100:end));
xtickangle(60);
xlim([new_time_sec(4*lengthT+1),new_time_sec(end)]);
title(categories{i});
end;
linkaxes([ax(1),ax(2),ax(3)],'x');
cd('C:/Users/eshchekinova/Documents/BoknisData/LSTMPred/Model');
new_time_sec1=new_time_sec(4*lengthT+1:lengthT*5);
X1=X(4*lengthT+1:lengthT*5,:);
new_time1=new_time(4*lengthT+1:lengthT*5);

clear X new_time_sec new_time
X=X1; 
new_time_sec=new_time_sec1;
new_time=new_time1;
save NormalizedBoknisShort new_time_sec X new_time categories