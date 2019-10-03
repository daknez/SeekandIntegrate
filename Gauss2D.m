function [Aall,Iall,Ires,indlist] = Gauss2D(I,peakpxx,peakpxy,LOC,BkgrdInt,minneighbourradius)

    waitb = waitbar(0,'Initialize Guassian fitting...');

    % additional Parameters:
    InterpMethod='nearest'; % 'nearest','linear','spline','cubic'
    FitOrientation='fit';	% 'fit': fit for orientation, 'dont' fit for orientation
        
    n = size(I,2);
    m = size(I,1);
    
    indlist = find(peakpxx>round(minneighbourradius) & peakpxx<m-round(minneighbourradius) & peakpxy>round(minneighbourradius) & peakpxy<n-round(minneighbourradius));
    
    % Numerical Grid
    [x,y]=meshgrid(1:m,1:n); X=zeros(m,n,2); X(:,:,1)=x; X(:,:,2)=y;
    
    % 1. 2D Gaussian function ( A requires 5 coefs ).
    % Amplitude, X-Coord, X-Width, Y-Coord, Y-Width
    g = @(A,X) A(1)*exp( -((X(:,:,1)-A(2)).^2/(2*A(3)^2) + (X(:,:,2)-A(4)).^2/(2*A(5)^2)) );

    % 2. 2D Rotated Gaussian function ( A requires 6 coefs ).
    % Amplitude, X-Coord, X-Width, Y-Coord, Y-Width, Angle
    f = @(A,X)  A(1)*exp( -(...
        ( X(:,:,1)*cos(A(6))-X(:,:,2)*sin(A(6)) - A(2)*cos(A(6))+A(4)*sin(A(6)) ).^2/(2*A(3)^2) + ... 
        ( X(:,:,1)*sin(A(6))+X(:,:,2)*cos(A(6)) - A(2)*sin(A(6))-A(4)*cos(A(6)) ).^2/(2*A(5)^2) ) );% + A(7);

    Aall=[];
    Iall = zeros(size(I));
    Ires = I;
    nopoints = length(indlist);
    
    for i = 1:nopoints

        waitbar(i/nopoints,waitb,['Fitting 2D Gaussians: ' num2str(i) '/' num2str(nopoints)]);

        index = indlist(i);

        Itemp=I-BkgrdInt;
        Itemp(LOC~=index)=0;

        [ytemp, xtemp] = find(Itemp~=0);
        xlo = min(xtemp);
        xhi = max(xtemp);
        ylo = min(ytemp);
        yhi = max(ytemp);
        Icut = Itemp(ylo:yhi,xlo:xhi);

        ncut = size(Icut,2);
        mcut = size(Icut,1);
                
        %figure
        %imagesc(Icut)

        %% ---Build numerical Grids---
        % Numerical Grid
        [x,y]=meshgrid(1:ncut,1:mcut); Xcut=zeros(mcut,ncut,2); Xcut(:,:,1)=x; Xcut(:,:,2)=y;
        % High Resolution Grid
        h=3; [xh,yh]=meshgrid(1/h:1/h:ncut,1/h:1/h:mcut); Xhcut=zeros(h*mcut,h*ncut,2); Xhcut(:,:,1)=xh; Xhcut(:,:,2)=yh;

        %% ---Fit---
        % Define lower and upper bounds [Amp,xo,wx,yo,wy,fi]
        
        peakintensity = max(max(Icut(round(mcut/4):mcut-round(mcut/4),round(ncut/4):ncut-round(ncut/4))));%max(Icut(:));%Icut(peakpxy(index)-ylo,peakpxx(index)-xlo)
        
%         lb = [peakintensity-2*sqrt(peakintensity),mcut/4,5,ncut/4,5,0,0];
%         ub = [peakintensity,mcut-mcut/4,minneighbourradius,ncut-ncut/4,minneighbourradius,pi/4,2*BkgrdInt];
        lb = [peakintensity-2*sqrt(peakintensity),mcut/4,5,ncut/4,5,0];
        ub = [peakintensity,mcut-mcut/4,minneighbourradius,ncut-ncut/4,minneighbourradius,pi/4];
        
        if i == 1   % initial guess:
            %A0 = double([peakintensity,peakpxx(index)-xlo,10,peakpxy(index)-ylo,10,0,BkgrdInt]);
            A0 = double([peakintensity,peakpxx(index)-xlo,10,peakpxy(index)-ylo,10,0]);
        else % use previous solution as initial guess:
            %A0 = double([peakintensity,peakpxx(index)-xlo,A(3),peakpxy(index)-ylo,A(5),0,A(7)]);
            A0 = double([peakintensity,peakpxx(index)-xlo,A(3),peakpxy(index)-ylo,A(5),0]);
        end
        
        % Options for solver:
        %opts = optimset('Algorithm','levenberg-marquardt','Display','off','UseParallel',true); 
        opts = optimset('Display','off'); 

        % Fit sample data
        switch FitOrientation
            case 'dont', [A,resnorm,res,flag,output] = lsqcurvefit(g,A0(1:5),Xcut,Icut,lb(1:5),ub(1:5),opts);
            case 'fit',  [A,resnorm,res,flag,output] = lsqcurvefit(f,A0,Xcut,Icut,lb,ub,opts);
            otherwise, error('invalid entry');
        end
        %disp(output); % display summary of LSQ algorithm

        A(2) = A(2)+xlo;
        A(4) = A(4)+ylo;
        
        Aall = [Aall; A];
        Iall = Iall + f(A,X);
        Ires=Ires-f(A,X);
    end
    
    Ires = Ires - BkgrdInt;
    delete(waitb);

end