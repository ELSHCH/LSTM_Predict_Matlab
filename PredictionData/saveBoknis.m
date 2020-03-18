%-----------------------------------------------------------------------------------------------
%      Save half of a given time series for Boknis Eck data to new time series       
%-----------------------------------------------------------------------------------------------
load('NormalizedBoknis.mat');
lengthT=floor(length(new_time_sec)/2); % define half of the time interval
% Copy first half of a given time series to the new time series
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
% Save the half of a given time series to new .mat file
save NormalizedBoknisHalf X new_time_sec new_time 