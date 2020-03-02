function R_covar = covarianceStateTransition(X_var_f,n_var_f,n_observ_f,net)
%------------------------------------------------------------------------
%    Estimate covariance of state transition for LSTM  
%    Input parameters: X_var_f - time series of size (n_observ_f X n_var_f);
%                      n_var_f - number of parameters;
%                      n_observ_f - number of time steps;
%    Output parameters: R_covar - matrix of covariance of state transition (n_var_f X n_var_f)
%    Last modified Elena 10/01/2020
%-------------------------------------------------------------------------
for i=1:n_observ_f
 [net,YPred] = predictAndUpdateState(net,X_var_f(i,1:n_var_f)');
  for j=1:n_var_f
    diff_var(i,j)=abs(X_var_f(i,j)-YPred(j));
  end;  
end; 
if n_var_f>1
   R_covar=zeros(n_var_f,n_var_f);
   for k = 1:n_var_f    
     mean_diff_var(k) = mean(diff_var(1:n_observ_f,k));  
   end;
   for i=1:n_var_f    
      for j=1:n_observ_f
         R_covar(i,i)=R_covar(i,i)+(X_var_f(j,i)- mean_diff_var(i))^2;         
      end     
   end
else
   R_covar=0;
   mean_diff_var = mean(diff_var_f(1:n_observ_f));
   for j=1:n_observ_f
         R_covar=R_covar+(diff_var_f(j)- mean_diff_var)^2;         
   end    
end  
R_covar=R_covar./(n_observ_f-1);