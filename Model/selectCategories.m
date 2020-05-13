function [nVar_f,categories_Pred,categories_Resp,ind_predictors,ind_responses,X_predictors_f]=selectCategories(namefile_f,categories_init_f,X_f,Var_f)
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
 k1=0;k2=0;
 while feof(fin)==0
  keyTrue_1=fscanf(fin,'%s\t',1);
  keyTrue_2=fscanf(fin,'%s\t',1);
  value=fscanf(fin,'%s\t',1);
  if strcmp(keyTrue_1,'1')==1
   k1=k1+1;   
   parameterPred{k1}=string(replace(value,'_',' '));
  end; 
  if strcmp(keyTrue_2,'1')==1
   k2=k2+1;   
   parameterResp{k2}=string(replace(value,'_',' '));
  end; 
 end;
 fclose(fin);
 lengthT=length(X_f(:,1));
 nVar_f=0;
 k=0;
% Select predictors according to defined and rearrange predictors with oxygen parameters being first
 for j=1:k1
  for i=1:Var_f
     categories_init_f{i}=replace(categories_init_f{i},'_',' ');
     if strcmp(parameterPred{j},cellstr(categories_init_f{i}))==1 
         nVar_f=nVar_f+1;
         categories_Pred{nVar_f}=categories_init_f{i};
         X_predictors_f(:,nVar_f)=X_f(:,i); 
         ind_predictors(nVar_f)=nVar_f;
     end;  
  end;
 end;
 k=0;
 for j=1:nVar_f
  for i=1:k2   
     if strcmp(categories_Pred{j},cellstr(parameterResp{i}))==1 
         k=k+1;
         categories_Resp{k}=cellstr(parameterResp{i});
         ind_responses(k)=j;
     end;  
  end;   
 end; 
 if k==0
    fprintf('Select correct parameter for prediction from\n');
    categories_Resp{1}='no categories';
    ind_responses(1)=0;
    return; 
 end;    