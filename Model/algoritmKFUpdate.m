function [P_next_f,Y_next_f]=algoritmKFUpdate(net,sampleSize,numpoints,ind_pred,nVar_f,Pred_last,Cov_last,X_tr)
%----------------------------------------------------------------------------------------------------------------------------------
%      Calculate prediction guess and covariance, covariance of model, covariance of data, Kalman Gain and update prediction point.    
%----------------------------------------------------------------------------------------------------------------------------------
%      Input parameters: net - trained LSTM network for prediction;
%                        sampleSize - size of ensembles used for estimation LSTM model covariance;     
%                        numpoints - length of fixed interval used for data, over sequence of size "numpoints" covariance of original data 
%                                    is estimated;
%                        ind_pred - index of time step for prediction; 
%                        nVar_f - number of parameters for prediction;  
%                        X_tr - original data; 
%     Output parameters: Y_next_f, P_next_f - prediction and covariance of prediction           
%
%         Last modified Elena 28/11/2019.  
%----------------------------------------------------------------------------------------------------------------------------------

for r1=1:sampleSize 
   [net,YPred_f] = predictAndUpdateState(net,Pred_last'); % Update prediction using last point of training series 
   Array_Pred(r1,1:nVar_f)=YPred_f(1:nVar_f);
end;
% Estimate prediction guess
for r1=1:nVar_f  
Y_Pred_f(r1)=mean(Array_Pred(:,r1)); 
end;

Q_cov_pred = sampleCovariance(Array_Pred,nVar_f,sampleSize); % estimate covariance of model

X_var(1:numpoints,1:nVar_f)=X_tr(ind_pred-numpoints+1:ind_pred,1:nVar_f); % from original data cut an interval of size 
%"numpoints" for estimation the covariance of sample
Q_covar = sampleCovariance(X_var,nVar_f,numpoints); % estimate covariance of underlying original data
  
%jwb = fpderiv('de_dwb',net,Pred_last,Y_Pred_f);
Jwb = jacobianCompute(Pred_last,Y_Pred_f); % estimate Jacobian of LSTM model
%P_pred_f = Jwb.*Cov_last.*transpose(Jwb)+Q_cov_pred; % estimate prediction covariance

P_pred_f = Q_cov_pred; % estimate prediction covariance
  
[K_F]=K_Gain(P_pred_f,Q_covar); % Estimate Kalman Gain
e = ones(nVar_f,1);
I_matrix=diag(e);
P_next_f=mtimes((I_matrix-K_F),P_pred_f); % update covariance of prediction

% Update initial prediction guess 
for r1=1:nVar_f
    sum=0;
 for r2=1:nVar_f   
  sum=sum+K_F(r1,r2)*(X_tr(ind_pred,r2)-Y_Pred_f(r2)); % update initial prediction guess 
 end;
  Y_next_f(r1)=Y_Pred_f(r1)+sum;
end; 