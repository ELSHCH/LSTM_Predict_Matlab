function [t_true_f,X_true_f,sig,mu,max_X_f,t_downscale_f,X_downscale_f,XTrain_f,YTrain_f,Nsteps_f] = ...
         PrepareData(n_points_f,P_horizon_sec_f,define_choice_training_f,TrainInter_f,...
                     n_Var_f,data_X_f,data_T_f)

%  Prepare data to standartized form according to selected training
%  interval
%
%  Use chickenpox_dataset available from Matlab  
%  Input parameters: 
%           n_points_f -  frequency for downscaling; 
%           P_horizon_sec_f - prediction horizon in seconds;
%           define_choice_training_f - can be 'FULL' or 'INTERVAL' for
%                                    training using entire length time or interval of
%                                    time_series correspondingly;
%           TrainInter_f - is length of trained interval;
%           n_Var_f - is dimension of parameters (1 - for single parameter, (>1) - for multiparameter study)
%           data_X_f - original data; 
%           data_T_f - time sequence; 
%  Output parameters:
%           t_true_f - time sequence;   
%           X_true_f - data sequence, copy of original time series; 
%           t_downscale_f - time sequences, reduced scale;  
%           X_downscale_f - data sequences, reduced scale; 
%           layers_f - layers for LSTM network;
%           options_f - parameters for LSTM network;
%           XTrain_f,YTrain_f  - sequences for predictors and  responses in
%                            correspondingly;
%           Nsteps_f - number of simulation time steps corresponding to given prediction horizon 
%      
%     Last modified Elena 26/11/2019 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

len_data=length(data_X_f(:,1));
m_points=floor(len_data/n_points_f);
X_downscale_f=zeros(m_points,n_Var_f);


for si=1:n_Var_f
% Standartize data

 mu(si) = mean(data_X_f(:,si));
 sig(si) = std(data_X_f(:,si));

 data_X_f(:,si)=(data_X_f(:,si)-mu(si))/sig(si);
 max_X_f(si)=max(data_X_f(:,si));
 data_X_f(:,si)=data_X_f(:,si)/max_X_f(si);

% Create copy of original time series 

 X_true_f(1,si)=data_X_f(1,si);
 t_true_f(1)=data_T_f(1);
 
 num_points=floor(len_data);
 for i=1:num_points
    X_true_f(i,si)=data_X_f(i,si);
    t_true_f(i)=data_T_f(i);
 end;

 deltaT=t_true_f(2)-t_true_f(1); % time step

 len_d=length(X_true_f(:,si)); % length of original time series

 Nsteps_f=floor(P_horizon_sec_f/deltaT); % number of time steps corresponing to prediction horizon ( here given in seconds)


 for i=1:m_points
     X_d_f(m_points-(i-1),si)=mean(data_X_f(1+len_data-i*n_points_f:len_data-(i-1)*n_points_f,si));
     t_d_f(m_points-(i-1))=data_T_f(len_data-(i-1)*n_points_f);
 end;
 
 t_downscale_f=linspace(t_d_f(1),t_d_f(end),m_points);

 X_downscale_f(:,si)=interp1(t_d_f,X_d_f(:,si),t_downscale_f);

% Assign data for training 

 if define_choice_training_f == "FULL" || TrainInter_f==1

% Entire time series is used for training
%    XTrain_f=zeros(len_data-1,n_Var_f);
%    YTrain_f=zeros(len_data-1,n_Var_f);
   XTrain_f(:,si) = X_downscale_f(1:end-1,si); % predictors
   YTrain_f(:,si) = X_downscale_f(2:end,si);   % responses

% Interval is used for training

 elseif define_choice_training_f == "PART"

   XTrain_f(:,si) = X_downscale_f(1:TrainInter_f-1,si);
   YTrain_f(:,si) = X_downscale_f(2:TrainInter_f,si);

 end;
end;
