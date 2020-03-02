function [x,t]=normData(xin,tin,ln)

%-----------------Interpolate data-----------------------
t=linspace(tin(1),tin(end),ln);
int_xin=interp1(tin,xin,t);
%--------------------Normalize data  -------------------
mean_xin = mean(int_xin);
int_xin=int_xin-mean_xin;
x = int_xin/norm(int_xin);