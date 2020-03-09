load('NormalizedBoknis.mat');
lengthT=floor(length(new_time_sec)/2);
for i=1:lengthT
  X_1(i,:)=X(i,:);
  new_time_sec_1(i)=new_time_sec(i);
  new_time_1(i)=new_time(i);
end;
clear X new_time_sec new_time
for i=1:lengthT
  X(i,:)=X_1(i,:);
  new_time_sec(i)=new_time_sec_1(i);
  new_time(i)=new_time_1(i);
end;
save NormalizedBoknisHalf X new_time_sec new_time 