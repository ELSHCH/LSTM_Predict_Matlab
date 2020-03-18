load('NormalizedBoknis.mat');
X_mean=zeros(floor(length(new_time)/20),1);
for i=1:floor(length(new_time)/20) 
X_mean(i)=X(i*20,7);
new_time100(i)=new_time_sec(i*20);
end;
%plot(new_time_sec,X(:,7),'r');
%hold on
plot(new_time100,cos(X_mean(:)));