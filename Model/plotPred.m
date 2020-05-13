load tr
for i=1:59
XX(i,1:325,1)=X{i}(1,1:325);
end;
for i=1:59
YY1(i,1:325,1)=XTrain{i}(1,1:325);
end;
for i=1:59
YY2(i,1:325,1)=YTrain{i}(1,1:325);
end;
hold on
for i=1:59
plot(1+325*i:325*(i+1),XX(i,1:325,1),'r');
end;
hold on
for i=1:59
plot(1+325*(i-1):325*i,YY1(i,1:325,1),'o');
end;
for i=1:59
plot(1+325*i:325*(i+1),YY2(i,1:325,1));
end;
%legend({'explained','predicted'});