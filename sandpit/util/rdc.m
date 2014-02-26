function r = rdc(x,y,k,s)
  if nargin <3
    k = 20;
  end
  if nargin <4
    s = 1/6;
  end
  n = size(x,1);
  x = [tiedrank(x)/n ones(n,1)];
  y = [tiedrank(y)/n ones(n,1)];
  x = sin(s/size(x,2)*x*randn(size(x,2),k));
  y = sin(s/size(y,2)*y*randn(size(y,2),k));
  warning('off','stats:canoncorr:NotFullRank')
  [~,~,r] = canoncorr([x ones(n,1)],[y ones(n,1)]);
  r = r(1);
