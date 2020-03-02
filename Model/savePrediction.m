function savePrediction(t_f,X_f,t_true,X_true,t_filter,X_filter,numResponses,cts,fileName,dirName)
%----------------------------------------------------------------------- 
% Save time and data for predicted, original and lowsampled/averaged time series
% in .mat and .dat formats
%------------------------------------------------------------------------
%   Input variables :
%         t_f, X_f - time and dat for predictions    
%         t_true, X_true - time and data for original data
%         t_filter, X_filter - filtered/lowsampled time and data 
%         fileName - name of output .mat file
%           
%   Last modified Elena 20/11/2019
%--------------------------------------------------------------------
cd(dirName);
X_f(1,1)
for i=1:numResponses
 categories(i)=cts(i);
end;
fileNameMat=strcat(fileName,'.mat');
save(fileNameMat,'t_f','X_f','t_true','X_true','t_filter','X_filter','categories');
% formatSpec = 'X is %4.2f meters or %8.3f mm\n';
file1=strcat(fileName,'.dat');
f=fopen(file1,'w');
for i=1:length(t_true)
 fprintf(f,'%g\t',t_true(i));   
 for j=1:numResponses-1   
  fprintf(f,'%g\t',X_true(i,j));
 end; 
 fprintf(f,'%g\n',X_true(i,numResponses));
end;
for i=1:length(t_filter)
 fprintf(f,'%g\t',t_filter(i));   
 for j=1:numResponses-1  
  fprintf(f,'%g\t',X_filter(i,j));
 end;
 fprintf(f,'%g\n',X_filter(i,numResponses));
end;
for i=1:length(t_f)
 fprintf(f,'%g\t',t_f(i));   
 for j=1:numResponses-1  
  fprintf(f,'%g\t',X_f(i,j));
 end;
 fprintf(f,'%g\n',X_f(i,numResponses));
end;
fclose(f);
