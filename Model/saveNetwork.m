function saveNetwork(net,dirNetwork,file_d)
%----------------------------------------------------------------------- 
% Save trained network in .mat format 
%------------------------------------------------------------------------
%   Input variables :
%         net - trained network   
%         dirNetwork - name of directory
%         num_points - frequency of sampling for filtered data series
%         choice_train - method for training , can be 'FULL' or 'INTERVAL'
%         file_d - name of network file
%           
%   Last modified Elena 20/11/2019
%--------------------------------------------------------------------
cd(dirNetwork);

save(file_d,'net');