function [ind_points_start,ind_points_end]=defineInterval(time_begin,t_original,t_low,end_step)
%---------------------------------------------------------------------------------
%    Definition of interval [t_low(ind_points_start),t_low(ind_points_end)] that overlap
%    with original interval [t_original(1+time_start),t_original(time_start+end_step]
%    Input variables: time_begin - start index for t_original sequence;
%                    end_step  - end index for t_original sequence;
%                    t_original - time sequence with original
%                    interpolation;
%                    t_low - filtered time sequence;
%    Output variables: ind_points_start - index for start for t_low sequence;
%                      ind_points_end - index for end of t_low sequence.    
%   
%  Last modified Elena 21.11.2019
%----------------------------------------------------------------------------------
gr1=0;
gr2=0;
ind_points=1;

% Find the start and end time indices in downscaled time series 't_low' that
% overlaps input prediction interval defined in original time series 't_original' 

for s1=1:length(t_low)
  if (t_low(s1)>=t_original(1+time_begin)) 
   if (gr1==0)  
      ind_points_start=s1;
      gr1=1;
    end;
   end; 
  if (t_low(s1)<=t_original(time_begin+end_step))   
      ind_points_end=s1;
    end;
end  