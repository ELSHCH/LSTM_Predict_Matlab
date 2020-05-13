function [Algorithm_Scheme_f,choice_training_f,time_start_f,P_horizon_f,n_points_f,nVar_f,train_f,sampleSize_f,fileData_f]=InputParam(fileName_f)
%-----------------------------------------------------------------------------------------------------------------
%               Initialize parameters for the LSTM algorithm from given .dat file
%-----------------------------------------------------------------------------------------------------------------
%  Input variables: fileName_f - name of .dat file with parameters defined
%                                by users;
%  Output variables: Algorithm_Scheme_f - algorithm scheme for LSTM prediction;
%                    choice_training_f - choice of training for LSTM prediction;
%                    LSTM_training_f - options for prediction;
%                    time_start_f - start time in format "2018-07-18 00:00:30";
%                    N_steps_f - number of prediction steps (seconds);
%                    batch_size_f - frequency for low rate sampling or time window for averging; 
%                    nVar_f - number of parameters used for prediction;
%                    fileData_f - name of data file 
%
% Last Modified Elena 26/11/2019
%-----------------------------------------------------------------------------------------------------------------

fin=fopen(fileName_f,'r');

Algorithm_Scheme_f=fscanf(fin,'%s\n',1); % three types of schemes are used 'LSTM STANDALONE',...
            % 'KALMAN FILTER STANDALONE','1 LSTM & KALMAN FILTER' and '3 LSTM & KALMAN FILTER'

choice_training_f=fscanf(fin,'%s\n',1); % two choices are used 'FULL','PART'

sampleSize_f=fscanf(fin,'%d\n',1); % length of training interval if 'INTERVAL' training is chosen

time_start_f=fscanf(fin,'%s\n',1); %start time in format "2018-07-18 00:00:30" for prediction     
time_start_f=replace(time_start_f,'_',' ')
P_horizon = fscanf(fin,'%d\n',1); % prediction horizon (in hours);

P_horizon_f = P_horizon*3600; % number of prediction steps (in seconds);
   
n_points_f=fscanf(fin,'%d\n',1); % frequency for low rate sampling/averaging of data;    

nVar_f=fscanf(fin,'%d\n',1); % number of parameters     

train_f=fscanf(fin,'%s\n',1); % train - for training option, test - for testing option    

fileData_f=fscanf(fin,'%s\n',1); % name of data file   

fclose(fin);
