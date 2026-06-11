function A = sym_array(A, sub_dim)
% partially symmetrize a high oder array
% sub_dim: a subset of {1, 2, ..., m}
m = ndims(A);
if ~exist('sub_dim', 'var')
    sub_dim = 1:m;
end
if iscell(sub_dim)
    for ii=1:length(sub_dim)
        A = sym_array(A, sub_dim{ii});
    end
    return
end
d = length(sub_dim);
m = ndims(A);

Asym = A;
for kk=d-1:-1:1
    for  jj=kk+1:d
        ns = 1:m;
        ns([sub_dim(kk), sub_dim(jj)]) = [sub_dim(jj), sub_dim(kk)];
        Asym = Asym + permute(A, ns);
    end
    A = Asym;
end
A = A/(factorial(d));
end
