function Q_covar = covariancePred(X_var_f,n_var_f,n_observ_f,net)
%------------------------------------------------------------------------
%    Estimate prediction covariance 
%    Input parameters: X_var_f - time series of size (n_observ_f X n_var_f);
%                      n_var_f - number of parameters in time
%                      series;
%                      n_observ_f - number of time steps;
%    Output parameters: Q_covar - matrix of prediction covariance of size (n_var_f X n_var_f)
%    Last modified Elena 29/01/2020
%-------------------------------------------------------------------------
for i=1:n_observ_f
 [net,Y_f] = predictAndUpdateState(net,X_var_f(end-1,1:n_var_f)');
 Ypred_f(i,1:n_var_f)=Y_f;
end;
if n_var_f>1
   Q_covar=zeros(n_var_f,n_var_f);
   for k = 1:n_var_f    
     mean_Ypred_f(k) = mean(Ypred_f(1:n_observ_f,k));  
   end;
   for i=1:n_var_f    
      for j=1:n_observ_f
         Q_covar(i,i)=Q_covar(i,i)+(Ypred_f(j,i)- mean_Ypred_f(i))^2;         
      end     
   end
else
    Q_covar=0;
   mean_Ypred_f = mean(Ypred_f(1:n_observ_f));
   for j=1:n_observ_f
         Q_covar=Q_covar+(Ypred_f(j)- mean_Ypred_f)^2;         
   end    
end  
Q_covar=Q_covar./(n_observ_f-1);