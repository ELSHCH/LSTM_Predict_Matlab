load LSTMNoUpdate
load LSTMUpdate
load LSTMUpdateOption1 
subplot(2,1,1);

plot(time,data,'Color','b','LineWidth',2)
hold on
plot(DataLSTMUpdate.t_f,DataLSTMUpdate.X_f,'Color',[0.9 0.1 0.4],'LineWidth',2)
plot(DataLSTMNoUpdate.t_f,DataLSTMNoUpdate.X_f,'Color',[0.5 0.7 0.9],'LineWidth',2)
plot(DataLSTMUpdateOption1.t_f,DataLSTMUpdateOption1.X_f,'Color',[0.1 0.9 0.1],'LineWidth',2)
legend('original data','forecast, algorithm with update','forecast, algorithm without update',...
    'forecast, algorithm with interval update')
subplot(2,1,2);
k=0;
for i=1:length(time)
for k=1:length(DataLSTMNoUpdate.X_f)
    if (time(i)==DataLSTMNoUpdate.t_f(k))
rmse_noUpdate(k)=abs(DataLSTMNoUpdate.X_f(k)-data(i));
t_noUpdate(k)=time(i);
end;
end;
end;
for i=1:length(time)
for k=1:length(DataLSTMUpdate.X_f)
    if (time(i)==DataLSTMUpdate.t_f(k))
rmse_Update(k)=abs(DataLSTMUpdate.X_f(k)-data(i));
t_Update(k)=time(i);
end;
end;
end;
for i=1:length(time)
for k=1:length(DataLSTMUpdateOption1.X_f)
    if (time(i)==DataLSTMUpdateOption1.t_f(k))
rmse_UpdateOption(k)=abs(DataLSTMUpdateOption1.X_f(k)-data(i));
t_UpdateOption(k)=time(i);
end;
end;
end;

mean(rmse_UpdateOption)
mean(rmse_Update)   % lesser
mean(rmse_noUpdate) % greater


plot(t_Update,rmse_Update,'Color',[0.9 0.1 0.4],'LineWidth',2);
hold on
plot(t_noUpdate,rmse_noUpdate,'Color',[0.5 0.7 0.9],'LineWidth',2);
plot(t_UpdateOption,rmse_UpdateOption,'Color',[0.1 0.9 0.1],'LineWidth',2);
legend('RMSE, algorithm with update','RMSE, algorithm with no update','RMSE, algorithm with update+ interval training')