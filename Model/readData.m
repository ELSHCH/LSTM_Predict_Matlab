function [data_X_f,data_T_f,data_T_date,ind_start]=readData(X_f,t_f,t_sec_f,time_start_f,nVar_f)
%  Convert time of prediction start to integer index and convert  
%                        time series to sequential array format 
%  Input parameters: X_f, t_f, t_sec_f - initial data, time in date/seconds format      
%                    time_start_f   - initial start of prediction in
%                                     date/integer format
%                    nVar_f - number of parameters ( number of time series)
%  Output parameters: data_X_f, data_T_f, data_T_date - copies of data, time series (date/seconds format) arrays 
%                     ind_start - start index corresponding to initial start of prediction 
% Last modified Elena 28/11/2019 
%-------------------------------------------------------------------------------------------------
base = datenum(1970,1,1);
if class(time_start_f)=='char'
  t_start=posixtime(datetime(time_start_f,'Format','dd-MMM-yyyy HH:mm:ss')); %convert string value to date
 % t_start(1),

  if t_start>t_sec_f(end)
   ind_start=0;
  elseif t_start<t_sec_f(1)
   ind_start=-1;   
  else    
   ind_f=find(t_sec_f>=t_start);
   ind_start=ind_f(1);
  end; 
elseif class(time_start_f)=='double'
  ind_start=time_start_f;  
end;

for i=1:length(t_f) 
 for si=1:nVar_f
   data_X_f(i,si) = X_f(i,si);
 end;  
   data_T_f(i)=t_sec_f(i);
   data_T_date(i)=t_f(i);
end;
