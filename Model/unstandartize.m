[X_f_mean,t_true,X_true,t_filter,X_filter]=un_std(mu,sig,max_D,X_f_mean_fun,X_true_fun,X_filter_fun,ind_responses) 
nVar_res=length(X_f_mean(1,:));

for si=1:nVar_res
  X_true(:,ind_response(si))=(X_true_fun(:,ind_response(si))+mu(ind_response(si)))*sig(ind_response(si))*max_D(ind_response(si));
  X_filter(:,ind_response(si))=(X_filter_fun(:,ind_response(si))+mu(ind_response(si)))*sig(ind_response(si))*max_D(ind_response(si));
  X_f_mean(:,si)=(X_f_mean_fun(:,si)+mu(ind_response(si)))*sig(ind_response(si))*max_D(ind_response(si));
end;