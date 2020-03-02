function [K]=K_Gain(Q_pred,Q_original)
%% Estimate Kalman Gain
 
 K=Q_pred.*(Q_pred+Q_original)^(-1); % Kalman Gain
