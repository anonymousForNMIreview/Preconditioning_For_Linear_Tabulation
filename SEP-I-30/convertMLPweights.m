% this script converts the f^{MLP} weights output by PyTorch for use in the
% C++/Fortran 90 implementation of f^{MLP}

D = dlmread('fc1w.csv');

D = D';

dlmwrite('A1.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc2w.csv');

D = D';

dlmwrite('A2.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc3w.csv');

D = D';

dlmwrite('A3.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc4w.csv');

D = D';

dlmwrite('A4.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc5w.csv');

D = D';

dlmwrite('A5.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc6w.csv');

D = D';

dlmwrite('A6.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc1b.csv');

D = D';

dlmwrite('B1.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc2b.csv');

D = D';

dlmwrite('B2.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc3b.csv');

D = D';

dlmwrite('B3.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc4b.csv');

D = D';

dlmwrite('B4.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc5b.csv');

D = D';

dlmwrite('B5.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = dlmread('fc6b.csv');

D = D';

dlmwrite('B6.csv',D, 'precision','%20.15f');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%