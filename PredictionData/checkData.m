clear all
fileData='NormalizedBoknis_new';
fileD=strcat(fileData,'.dat');
fl=fopen(fileD,'r');
%categ=fscanf(fl,'%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n');
dd= textscan(fl,'%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n');
a = cellstr(dd{1});
X=zeros(length(a)-1,length(dd)-1);
size(X)
for i=1:length(dd)
a= cellstr(dd{i});
categories{i}=string(a(1));
 for j=2:length(a)     
   if i==1
     new_time_sec(j-1)=str2num(string(a(j)));
   else  
     X(j-1,i-1)=str2num(string(a(j)));
   end;  
 end;
end; 
size(new_time_sec)
plot(new_time_sec,X(:,1));
