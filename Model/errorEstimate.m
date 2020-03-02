function [rmse_f,smape_f,mae_f,r_sqrt_f,MedAE_f,accur_f,prec_f,recall_f,F1_f]=errorEstimate(t_true_f,X_true_f,t_forecast_f,X_forecast_f)
%----------------------------------------------------------------------------------------------------------------
%               Evaluate prediction quality using standard diagnistic metrices
%----------------------------------------------------------------------------------------------------------------
%  Input parameters: t_true_f, X_true_f - original time and data
%  Output_parameters: rmse_f - root mean squared error;                   
%                     smape_f - symmetric mean absolute perecentage error;
%                     r_saqrt_f - coefficient of determination;  
%                     mae_f - mean absoulte error; 
%                     MedAE_f - regression metrics; 
%                     accur_f - accuracy;
%                     prec_f - precision;
%                     recall_f - recall;
%                     F1_f - classification scores;
%
%  See Ref.Ga´bor Petneha´zi, "Recurrent Neural Networks for Time Series Forecasting" 2019 (arXiv:1901.00069v1)
%   for definition of standard diagnistic metrices used here 
%
% Last modified 26/11/2019 Elena 
%----------------------------------------------------------------------------------------------------------------

length_interval=length(t_forecast_f);
l_TimeSeries=length(t_true_f);
X_observ(1)=X_true_f(1);
diff_X_true(1)=X_true_f(1);
for i=2:length_interval
  X_observ(i)=X_true_f(i);
  diff_X_true(i)=abs(X_true_f(i)-X_true_f(i-1));
end;    
min_diff=min(diff_X_true);
sumSquared=sum((X_observ(:)-X_forecast_f(:)).^2);
Matr_inclin_true=zeros(length_interval,1);
Matr_inclin_forecast=zeros(length_interval,1);
abs_diffX(1)=abs(X_observ(1)-X_forecast_f(1));
for i=2:length_interval
 cond_inclin_true(i)=cast(X_observ(i)-X_observ(i-1)>=0,'uint8');
 cond_inclin_forecast(i)=cast(X_forecast_f(i)-X_forecast_f(i-1)>=0,'uint8');
 abs_diffX(i)=abs(X_observ(i)-X_forecast_f(i));
end;
%Root mean squared error
rmse_f=sqrt(1/length_interval*(sumSquared));
%Symmetric mean absolute perecentage error
smape_f=0;
for i=1:length_interval
  smape_f=smape_f+abs(X_observ(i)-X_forecast_f(i))/(abs(X_observ(i))+abs(X_forecast_f(i)))/2;
end;
smape_f=100/length_interval*smape_f;
% Coefficient of determination
r_sqrt_f=0;
for i=1:length_interval
  r_sqrt_f=r_sqrt_f+(X_observ(i)-X_forecast_f(i))^2/(abs(X_observ(i))+abs(X_forecast_f(i)))/2;
end;
r_sqrt_f=1-100/length_interval*r_sqrt_f;
% Mean absoulte error
mae_f=1/length_interval*sum(abs(X_observ(:)-X_forecast_f(:)));
% Regression metrics 
MedAE_f=median(abs_diffX);
% Accuracy
accur_f=1/l_TimeSeries*double(condTrueEqual(cond_inclin_true,cond_inclin_forecast));
% accuracy = (TruePositive+TrueNegative)/TruePositive+TrueNegative+FalsePositive+FalseNegative);
% Precision
prec_f=double(condTwoEqualOne(cond_inclin_true,cond_inclin_forecast))/double(condOneEqualOne(cond_inclin_forecast)); % precision=TruePositive/(TruePositive+falsePositive)
% Recall
recall_f=double(condTwoEqualOne(cond_inclin_true,cond_inclin_forecast))/double(condOneEqualOne(cond_inclin_true)); %recall=TruePositive/(TruePositive+FalseNegative);
% Classification scores
F1_f=2*prec_f*recall_f/(prec_f+recall_f);
%% Supplementary methods 
function f_equal=condTrueEqual(cond_inclin_true,cond_inclin_forecast)
 f_equal=0;
for i=1:length(cond_inclin_true)
 f_equal=double(f_equal)+double(cast(cond_inclin_true(i)-cond_inclin_forecast(i)==0,'uint8'));
end;
function f_equal=condTwoEqualOne(cond_inclin_true,cond_inclin_forecast);
 f_equal=0;
for i=1:length(cond_inclin_true)
 f_equal=double(f_equal)+double(cast((cond_inclin_true(i)==1&&cond_inclin_forecast(i)==1),'uint8'));
end; 
function f_equal=condOneEqualOne(cond_inclin_forecast);
 f_equal=0;
for i=1:length(cond_inclin_forecast)
 f_equal=double(f_equal)+double(cast(cond_inclin_forecast(i)==1,'uint8'));
end; 