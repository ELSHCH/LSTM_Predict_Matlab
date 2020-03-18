function R_covar = covarianceStateTransitionPattern(X_var_f,X_Pred_f,n_var_f,n_observ_f)
%------------------------------------------------------------------------
%    Estimate covariance of state transition for LSTM  
%    Input parameters: X_var_f - time series of size (n_observ_f X n_var_f);
%                      n_var_f - number of response parameters;
%                      n_observ_f - number of time steps;
%    Output parameters: R_covar - matrix of covariance of state transition (n_var_f X n_var_f)
%    Last modified Elena 20/02/2020
%-----------------------------------------------------------------------

for i=1:n_var_f
  for j=1:n_observ_f
    diff_var(i,j)=abs(X_var_f(i,j)-X_Pred_f(i,j)); 
  end;
end;  
if n_var_f>1
   R_covar=zeros(n_var_f,n_var_f);
   for k = 1:n_var_f    
     mean_diff_var(k) = mean(diff_var(k,1:n_observ_f));  
   end;
   for i=1:n_var_f    
      for j=1:n_observ_f
         R_covar(i,i)=R_covar(i,i)+(X_var_f(i,j)- mean_diff_var(i))^2;         
      end     
   end
else
   R_covar=0;
   mean_diff_var = mean(diff_var(1:n_observ_f));
   for j=1:n_observ_f
         R_covar=R_covar+(diff_var(j)- mean_diff_var)^2;         
   end    
end  
R_covar=R_covar./(n_observ_f-1);