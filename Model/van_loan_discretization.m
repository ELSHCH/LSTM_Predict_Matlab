function [sigma, Q]=van_loan_discretization(F, G, dt)
%-----------------------------------------------------------------------------
% Discretizes a linear differential equation which includes white noise
% according to the method of C. F. van Loan [1]. Given the continuous
% model
%
%            x' =  Fx + Gu
%
%        where u is the unity white noise, we compute and return the sigma and Q_k
%        that discretizes that equation.
%-------------------------------------------------------------------------------
%
%        Examples
%        --------
%
%            Given y'' + y = 2u(t), we create the continuous state model
%            of
%
%            x' = [ 0 1] * x + [0]*u(t)
%                 [-1 0]       [2]
%
%            and a time step of 0.1:
% 
%  F = np.array([[0,1],[-1,0]], dtype=float)
%             >>> G = np.array([[0.],[2.]])
%             >>> phi, Q = van_loan_discretization(F, G, 0.1)
% 
%             >>> phi
%             array([[ 0.99500417,  0.09983342],
%                    [-0.09983342,  0.99500417]])
% 
%             >>> Q
%             array([[ 0.00133067,  0.01993342],
%                    [ 0.01993342,  0.39866933]])
% 
%             (example taken from Brown[2])
% 
% 
%         References
%         ----------
% 
%         [1] C. F. van Loan. "Computing Integrals Involving the Matrix Exponential."
%             IEEE Trans. Automomatic Control, AC-23 (3): 395-404 (June 1978)
% 
%         [2] Robert Grover Brown. "Introduction to Random Signals and Applied
%             Kalman Filtering." Forth edition. John Wiley & Sons. p. 126-7. (2012)
%         """
%-------------------------------------------------------------------------------------------------

 n = length(F(:,1));
 A = zeros(2*n, 2*n);

 % we assume u(t) is unity, and require that G incorporate the scaling term
 % for the noise. Hence W = 1, and GWG' reduces to GG"


 A(1:n,1:n) = -F*dt;
 A(1:n,n+1:2*n) = G*transpose(G)*dt;
 A(n+1:2*n, n+1:2*n) = transpose(F)*dt;

 B=expm(A);

 sigma = transpose(B(n+1:2*n,n+1:2*n));

 Q = sigma*B(1:n, n+1:2*n);

