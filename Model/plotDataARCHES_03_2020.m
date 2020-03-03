cd('C:/Users/eshchekinova/Documents/BoknisData/LSTMPred/PredictionData');
load('NormalizedBoknis.mat');
%load('NormalizedBoknis26-Sep-2019.mat');
for i=1:length(new_time)
m_r=datestr(new_time(i));
dm(i)=cellstr(m_r(1:11));
end;
lengthT=floor(length(new_time)/2);

ax(1)=subplot(4,1,1);
plot(ax(1),new_time_sec,X(:,2));
hold on
plot(ax(1),new_time_sec(1:lengthT),X(1:lengthT,2),'r');
xticks(new_time_sec(1:2000:end));
xticklabels(dm(1:2000:end));
xtickangle(60);
xlim([new_time_sec(1),new_time_sec(end)]);
title(categories{2},'FontSize',12);

ax(2)=subplot(4,1,2);
plot(ax(2),new_time_sec,X(:,1));
hold on
plot(ax(2),new_time_sec(1:lengthT),X(1:lengthT,1),'r');
xticks(new_time_sec(1:2000:end));
xticklabels(dm(1:2000:end));
xtickangle(60);
xlim([new_time_sec(1),new_time_sec(end)]);
title(categories{1},'FontSize',12);

ax(3)=subplot(4,1,3);
plot(ax(3),new_time_sec,X(:,8));
hold on
plot(ax(3),new_time_sec(1:lengthT),X(1:lengthT,8),'r');
xticks(new_time_sec(1:2000:end));
xticklabels(dm(1:2000:end));
xtickangle(60);
xlim([new_time_sec(1),new_time_sec(end)]);
title(categories{8},'FontSize',12);

ax(4)=subplot(4,1,4);
plot(ax(4),new_time_sec,X(:,9));
hold on
plot(ax(4),new_time_sec(1:lengthT),X(1:lengthT,9),'r');
xticks(new_time_sec(1:2000:end));
xticklabels(dm(1:2000:end));
xtickangle(60);
xlim([new_time_sec(1),new_time_sec(end)]);
title(categories{9},'FontSize',12);

linkaxes([ax(1),ax(2),ax(3),ax(4)],'x');
cd('C:/Users/eshchekinova/Documents/BoknisData/LSTMPred/Model');