function [Aall,Iall,Ires,indlist] = Gauss2D(I,peakpxy,peakpxx,LOC,BkgrdInt,minneighbourradius,FitOrientation)

    waitb = waitbar(0,'Initialize Guassian fitting...');

    % additional Parameters:
%     FitOrientation='dont';	% 'fit': fit for orientation, 'dont' fit for orientation

    [m,n] = size(I);
    
    indlist = find(peakpxx>round(minneighbourradius) & peakpxx<m-round(minneighbourradius) & peakpxy>round(minneighbourradius) & peakpxy<n-round(minneighbourradius));
    max(peakpxx)
    max(peakpxy)
    % Numerical Grid
    [x,y]=meshgrid(1:n,1:m); 
    X=zeros(m,n,2); 
    X(:,:,1)=x; 
    X(:,:,2)=y;
    
    % 1. 2D Gaussian function ( A requires 5 coefs ).
    % Amplitude, X-Coord, X-Width, Y-Coord, Y-Width
    g1 = @(A,X) A(1)*exp( -((X(:,:,1)-A(2)).^2/(2*A(3)^2) + (X(:,:,2)-A(4)).^2/(2*A(5)^2)) );
    
    
    g2 = @(A,X) A(1)*exp( -((X(:,:,1)-A(2)).^2/(2*A(3)^2) + (X(:,:,2)-A(4)).^2/(2*A(3)^2)) );

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

        [ytemp, xtemp] = find(Itemp~=0)
        xlo = min(xtemp);
        xhi = max(xtemp);
        ylo = min(ytemp);
        yhi = max(ytemp);

        Icut = Itemp(ylo:yhi,xlo:xhi);
        
        [mcut,ncut] = size(Icut);

        %% ---Build numerical Grids---
        % Numerical Grid
        [x,y]=meshgrid(1:ncut,1:mcut); 
        Xcut=zeros(mcut,ncut,2); 
        Xcut(:,:,1)=x; 
        Xcut(:,:,2)=y;
%         
%         % High Resolution Grid
%         h=3; 
%         [xh,yh]=meshgrid(1/h:1/h:ncut,1/h:1/h:mcut); 
%         Xhcut=zeros(h*mcut,h*ncut,2); 
%         Xhcut(:,:,1)=xh; 
%         Xhcut(:,:,2)=yh;

        %% ---Fit---
        % Define lower and upper bounds [Amp,xo,wx,yo,wy,fi]
        
        peakintensity = max(max(Icut(round(mcut/5):mcut-round(mcut/5),round(ncut/5):ncut-round(ncut/5))));%max(Icut(:));%Icut(peakpxy(index)-ylo,peakpxx(index)-xlo)
        
%         lb = [peakintensity-2*sqrt(peakintensity),mcut/5,5,ncut/5,5,0,0];
%         ub = [peakintensity,mcut-mcut/5,minneighbourradius,ncut-ncut/5,minneighbourradius,pi/5,2*BkgrdInt];
        lb = [peakintensity-2*sqrt(peakintensity),mcut/5,minneighbourradius/10,ncut/5,minneighbourradius/10,0];
        ub = [peakintensity+2*sqrt(peakintensity),mcut-mcut/5,minneighbourradius*2,ncut-ncut/5,minneighbourradius*2,pi/4];
        
        if i == 1   % initial guess:
            %A0 = double([peakintensity,peakpxx(index)-xlo,10,peakpxy(index)-ylo,10,0,BkgrdInt]);
            A0 = double([peakintensity,peakpxx(index)-xlo,10,peakpxy(index)-ylo,10,0]);
        else % use previous solution as initial guess:
            %A0 = double([peakintensity,peakpxx(index)-xlo,A(3),peakpxy(index)-ylo,A(5),0,A(7)]);
            A0 = double([peakintensity,peakpxx(index)-xlo,A(3),peakpxy(index)-ylo,A(3),0]);
        end
        
        % Options for solver:
        %opts = optimset('Algorithm','levenberg-marquardt','Display','off','UseParallel',true); 
        opts = optimset('Display','off'); 

        % Fit sample data
        switch FitOrientation
            case 'dont' 
                [A,resnorm,res,flag,output] = lsqcurvefit(g2,A0(1:4),Xcut,Icut,lb(1:4),ub(1:4),opts);
                A(2) = A(2)+xlo-1;
                A(4) = A(4)+ylo-1;
                Iall = Iall + g2(A(1:4),X);
                Ires=Ires-g2(A(1:4),X);
                
                A(5) = A(3);    % sig1=sig2
                A(6) = 0;   % no angle fitted
                Aall = [Aall; A];
            case 'fit2' 
                [A,resnorm,res,flag,output] = lsqcurvefit(g1,A0(1:5),Xcut,Icut,lb(1:5),ub(1:5),opts);
                A(2) = A(2)+xlo-1;
                A(4) = A(4)+ylo-1;
                Iall = Iall + g1(A(1:5),X);
                Ires=Ires-g1(A(1:5),X);
                Aall = [Aall; A(1:5)];
            case 'fit1'
                [A,resnorm,res,flag,output] = lsqcurvefit(f,A0,Xcut,Icut,lb,ub,opts);
                A(2) = A(2)+xlo-1;
                A(4) = A(4)+ylo-1;
                Iall = Iall + f(A,X);
                Ires=Ires-f(A,X);
                Aall = [Aall; A];
            otherwise, error('invalid entry');
        end
        %disp(output); % display summary of LSQ algorithm
        
%        %if mod(i,50) == 0
%        if i<=5
%             figure
%             imagesc(Icut)
%             
%             figure
%             imagesc(g2([A(1),A(2)-xlo,A(3),A(4)-ylo],Xcut))
%         end

    end
    
    Ires = Ires - BkgrdInt;
    delete(waitb);

end