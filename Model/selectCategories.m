function [nVar_f,X_predictors_f]=selectCategories(namefile_f,categories_init_f,X_f,Var_f)
%------------------------------------------------------------------------------------------
%   select parameters from file with list of all parameters and form an array
%   of predictors based on selected categories
%------------------------------------------------------------------------------------------
%   Input variables: namefile_f - name of file with list of categories; 
%                    categories_init_f - list of intial categories of
%                    predictors;
%                    X_f - initial array of predictors;
%                    Var_f - initial number of variables; 
%   Output variables: X_predictors_f - final array of predictors;
%                    nVar_f - number of selected predictors;
%------------------------------------------------------------------------------------------
% Last modified 01/03/2020 E. Shchekinova
%------------------------------------------------------------------------------------------
 fin=fopen(namefile_f,'r');
 k=0;
 while feof(fin)==0
  keyTrue=fscanf(fin,'%s\t',1);
  value=fscanf(fin,'%s\t',1);
  if strcmp(keyTrue,'1')==1
   k=k+1;   
   parameterDef{k}=strrep(value,'_',' ');
  end; 
 end;
 fclose(fin);
 lengthT=length(X_f(:,1));
 nVar_f=0;
% Select predictors according to defined 
 for j=1:k
 for i=1:Var_f
     if strcmp(cellstr(parameterDef{j}),cellstr(categories_init_f{i}))==1 
         nVar_f=nVar_f+1;
         X_predictors_f(:,nVar_f)=X_f(:,i);
     end;  
   end;   
 end; 