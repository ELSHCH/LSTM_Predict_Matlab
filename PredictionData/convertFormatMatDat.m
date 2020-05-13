% load NormalizedBoknis
% % file1=fopen('InitialCategories.dat','w');
% % for i=1:12
% %   categories{i}=replace(categories{i},' ','_');
% %   categories{i}
% %   fprintf(file1,'%s\n',categories{i});
% % end; 
% % fclose(file1);
% file1=fopen('NormalizedBoknis.dat','w');
% formatOut = 'mm/dd/yy hh:MM:ss';
%  for i=1:length(new_time_sec)
%  %    i
%  %  ti=datestr(new_time,formatOut);
%    fprintf(file1,'%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n',...
%         new_time_sec(i),X(i,1),X(i,2),X(i,3),X(i,4),X(i,5),X(i,6),...
%         X(i,7),X(i,8),X(i,9),X(i,10),X(i,11),X(i,12));  
% %fprintf(file1,'%g\n',new_time_sec(i));
%  end;
% %fprintf(file1,'%g',new_time_sec,[1, inf])
% fclose(file1);

%load NormalizedBoknis_3LSTM_PATTERN_KF_FULL_100_12_2_3
load NormalizedBoknis
% file1=fopen('InitialCategories.dat','w');
for i=1:12
  categories{i}=replace(categories{i},' ','_');
  categories{i}
end; 
% fclose(file1);
% file4=fopen('NormalizedBoknis.dat','r');
% data=fscanf(file4,'%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n',[13, inf]);
% fclose(file4);
% data(1,1)-data(1,2)
file4=fopen('NormalizedBoknis_new.dat','w');
fprintf(file4,'%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n',"time",categories{1},categories{2},categories{3},categories{4},categories{5},...
    categories{6},categories{7},categories{8},categories{9},categories{10},categories{11},categories{12});
for i=1:length(new_time_sec)
 %    i
 %  ti=datestr(new_time,formatOut);
   fprintf(file4,'%14.0f\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n',new_time_sec(i),X(i,1),X(i,2),X(i,3),X(i,4),X(i,5),X(i,6),X(i,7),...
       X(i,8),X(i,9),X(i,10),X(i,11),X(i,12));  
%fprintf(file1,'%g\n',new_time_sec(i));
end;
fclose(file4);
% 
% file1=fopen('Pred.dat','w');
% file2=fopen('True.dat','w');
% file3=fopen('Filter.dat','w');
% %formatOut = 'mm/dd/yy hh:MM:ss';
% fprintf(file1,'%s\t%s\n','Time','Oxygen'); 
% for i=1:length(t_f)
%  %    i
%  %  ti=datestr(new_time,formatOut);
%    fprintf(file1,'%10.0f\t%g\n',t_f(i),X_f(i,1));  
% %fprintf(file1,'%g\n',new_time_sec(i));
% end;
% fclose(file1);
% fprintf(file3,'%s\t%s\t%s\t%s\n','Time','Oxygen','Wind_GEOMAR','Wind_Dir_GEOMAR'); 
% for i=1:length(t_filter)
%  %    i
%  %  ti=datestr(new_time,formatOut);
%    fprintf(file3,'%10.0f\t%g\t%g\t%g\n',t_filter(i),X_filter(i,1),X_filter(i,2),X_filter(i,3));  
% %fprintf(file1,'%g\n',new_time_sec(i));
% end;
% fclose(file3);
% fprintf(file2,'%s\t%s\t%s\t%s\n','Time','Oxygen','Wind_GEOMAR','Wind_Dir_GEOMAR'); 
% for i=1:length(t_true)
%  %    i
%  %  ti=datestr(new_time,formatOut);
%    fprintf(file2,'%10.0f\t%g\t%g\t%g\n',t_true(i),X_true(i,1),X_true(i,2),X_true(i,3)); 
% %fprintf(file1,'%g\n',new_time_sec(i));
% end;
% %fprintf(file1,'%g',new_time_sec,[1, inf])
% fclose(file2);