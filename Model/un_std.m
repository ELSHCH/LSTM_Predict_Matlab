function [X_f_mean,X_true,X_filter]=un_std(mu,sig,max_D,X_f_mean,X_true,X_filter,ind_responses) 
nVar_res=length(X_f_mean(1,:));

for si=1:nVar_res
  X_true(:,ind_responses(si))=(X_true(:,ind_responses(si))+mu(ind_responses(si)))*sig(ind_responses(si))*max_D(ind_responses(si));
  X_filter(:,ind_responses(si))=(X_filter(:,ind_responses(si))+mu(ind_responses(si)))*sig(ind_responses(si))*max_D(ind_responses(si));
  X_f_mean(:,si)=(X_f_mean(:,si)+mu(ind_responses(si)))*sig(ind_responses(si))*max_D(ind_responses(si));
end;