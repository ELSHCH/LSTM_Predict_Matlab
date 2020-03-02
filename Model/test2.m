a=input('define a: \n');

try
    x=a^2;
catch isnan(a)==1
    warning('a is not defined');
    a=input('define a: \n');
end