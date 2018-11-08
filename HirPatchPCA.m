function [Iden,Ires] = HirPatchPCA(I, startw, PCn, ansctrans)

    w = startw;
    minpatchno = 2;
    ro = size(I,1);
    
    if ansctrans
        I = 2.*sqrt(double(I)+3/8);
        disp('Anscombe transform performed')
    end

    index = 1;
    tempIden = zeros(size(I));
    tempIres = zeros(size(I));
    Iden = zeros(size(I));
    Ires = zeros(size(I));
    while ro/w > minpatchno
        [tempIden,tempIres] = patchpca(I,1,PCn,w);
        Iden = Iden + tempIden;
        Ires = Ires + tempIres;
        w=w*2;
        index = index+1;
    end
    Iden = Iden./(index-1);
    Ires = Ires./(index-1);

    if ansctrans
        Iden = Iden.^2./4+sqrt(3/2)./(4.*Iden)-11./(8.*Iden.^2)+5.*sqrt(3/2)./(8.*Iden.^3)-1/8;
        Ires = Ires.^2./4+sqrt(3/2)./(4.*Ires)-11./(8.*Ires.^2)+5.*sqrt(3/2)./(8.*Ires.^3)-1/8;
        disp('Inverse Anscombe transform performed')
    end
end