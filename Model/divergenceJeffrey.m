function D_J = divergenceJeffrey(X1,X2)
%--------------------------------------------------------------------------
% Jeffrey's divergence as measure of the prediction performance,
% calculated from two distributions of predicted and original time series
%    Input variables X1, X2: two time series 
%    Output variables D_J - Jeffrey's divergence
%
%--------------------------------------------------------------------------
% Jeffrey's divergence is calculated according to formulas:
%
%  distr_KullbackLeibler(X1 | X2)= log(std(X2) /std(X1)) +1/2( (std(X1)^2+(mean(X1)-mean(X2))^2)/std(X2)^2)  -1/2 
%  
%  distr_Jeffrey(X1 | X2)=distr_KullbackLeibler(X1 | X2)+ distr_KullbackLeibler(X2 | X1)
%  Ref. Kelvil Loh et al arXiv:1802.05141v2
%
%  Last modified E. Shchekinova 15.03.2020 
%--------------------------------------------------------------------------
nVar=length(X1(1,:)); % number of parameters
nObserv=length(X1(:,1)); % number of time points
run_points=2; % the mean and std estimated over fixed window size run_points 
mm=floor(nObserv/run_points);
sigmaP=zeros(nObserv,nVar);
sigmaQ=zeros(nObserv,nVar);
meanP=zeros(nObserv,nVar);
meanQ=zeros(nObserv,nVar);
D_KL_PQ=zeros(nObserv,nVar);
D_KL_QP=zeros(nObserv,nVar);
D_J=zeros(nObserv,nVar);
for ss=1:nVar
 for jj=2:nObserv-1    
  sigmaP(jj,ss)=std(X1(jj-1:jj+1,ss));
  sigmaQ(jj,ss)=std(X2(jj-1:jj+1,ss));
  meanP(jj,ss)=mean(X1(jj-1:jj+1,ss));
  meanQ(jj,ss)=mean(X2(jj-1:jj+1,ss));
  if sigmaQ(jj,ss)~=0 && sigmaP(jj,ss)~=0
   D_KL_PQ(jj,ss)=log(sigmaP(jj,ss)/sigmaQ(jj,ss))+...
    1/2*(sigmaP(jj,ss)^2+(meanP(jj,ss)-meanQ(jj,ss))^2)/sigmaQ(jj,ss)^2-1/2;
   D_KL_QP(jj,ss)=log(sigmaQ(jj,ss)/sigmaP(jj,ss))+...
     1/2*(sigmaQ(jj,ss)^2+(meanQ(jj,ss)-meanP(jj,ss))^2)/sigmaP(jj,ss)^2-1/2;
  end;
  D_J(jj,ss)=D_KL_PQ(jj,ss)+D_KL_QP(jj,ss);
 end;
end;
