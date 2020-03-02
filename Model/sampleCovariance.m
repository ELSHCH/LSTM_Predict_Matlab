function Q_covar = sampleCovariance(X_var_f,n_var_f,n_observ_f)
%------------------------------------------------------------------------
%    Estimate covariance matrix
%    Input parameters: X_var_f - time series of size (n_observ_f X n_var_f);
%                      n_var_f - number of parameters in time
%                      series;
%                      n_observ_f - number of time steps;
%    Output parameters: Q_covar - matrix of covariance of size (n_var_f X n_var_f)
%    Last modified Elena 26/11/2019
%-------------------------------------------------------------------------

if n_var_f>1
   Q_covar=zeros(n_var_f,n_var_f);
   for k = 1:n_var_f    
     mean_X_var(k) = mean(X_var_f(1:n_observ_f,k));  
   end;
   for i=1:n_var_f    
      for j=1:n_observ_f
         Q_covar(i,i)=Q_covar(i,i)+(X_var_f(j,i)- mean_X_var(i))^2;         
      end     
   end
else
    Q_covar=0;
   mean_X_var = mean(X_var_f(1:n_observ_f));
   for j=1:n_observ_f
         Q_covar=Q_covar+(X_var_f(j)- mean_X_var)^2;         
   end    
end    
Q_covar=Q_covar./n_observ_f;