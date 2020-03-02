function jacobian_matrix=jacobianCompute(sequenceInput,sequenceOutput)  
%----------------------------------------------------------------------------------------------------------------------------------
%      Calculate Jacobian of network model    
%----------------------------------------------------------------------------------------------------------------------------------
%      Input parameters: sequenceOutput - output of network,
%                        sequenceInput  - input of network;
%     Output parameters: jacobian_matrix - jacobian of network evaluated at            
%
%         Last modified Elena 12/12/2019.  
%----------------------------------------------------------------------------------------------------------------------------------
output_size=length(sequenceOutput);
input_size=length(sequenceInput);
for m=1:output_size
     for j=1:input_size
        grad_func = gradient(sequenceOutput(m),sequenceInput(j));
        jacobian_matrix(m,j)=grad_func;
     end;
end;     