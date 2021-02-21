function [Aall,Iall,Ires,indlist,resnorm_all] = Gauss2D(I,peakpxy,peakpxx,LOC,BkgrdInt,minneighbourradius,FitOrientation)

    waitb = waitbar(0,'Initialize Guassian fitting...');

    % additional Parameters:
%     FitOrientation='dont';	% 'fit': fit for orientation, 'dont' fit for orientation

    [m,n] = size(I);
    
    indlist = find(peakpxx>round(minneighbourradius/2) & peakpxx<m-round(minneighbourradius/2) & peakpxy>round(minneighbourradius/2) & peakpxy<n-round(minneighbourradius/2));
    max(peakpxx)
    max(peakpxy)
    % Numerical Grid
    [x,y]=meshgrid(1:n,1:m); 
    X=zeros(m,n,2); 
    X(:,:,1)=x; 
    X(:,:,2)=y;
    
    % 1. 2D Gaussian function ( A requires 5 coefs ).
    % Amplitude, X-Coord, X-Width, Y-Coord, Y-Width
    g = @(A,X) A(1)*exp( -((X(:,:,1)-A(2)).^2/(2*A(3)^2) + (X(:,:,2)-A(4)).^2/(2*A(5)^2)) );
    
    % sig1 = sig2:
    g1 = @(A,X) A(1)*exp( -((X(:,:,1)-A(2)).^2/(2*A(3)^2) + (X(:,:,2)-A(4)).^2/(2*A(3)^2)) );
    
    % with constant Offset:
    g2 = @(A,X) A(5) + A(1)*exp( -((X(:,:,1)-A(2)).^2/(2*A(3)^2) + (X(:,:,2)-A(4)).^2/(2*A(3)^2)) );

    % 2. 2D Rotated Gaussian function ( A requires 6 coefs ).
    % Amplitude, X-Coord, X-Width, Y-Coord, Y-Width, Angle
    f = @(A,X)  A(1)*exp( -(...
        ( X(:,:,1)*cos(A(6))-X(:,:,2)*sin(A(6)) - A(2)*cos(A(6))+A(4)*sin(A(6)) ).^2/(2*A(3)^2) + ... 
        ( X(:,:,1)*sin(A(6))+X(:,:,2)*cos(A(6)) - A(2)*sin(A(6))-A(4)*cos(A(6)) ).^2/(2*A(5)^2) ) );% + A(7);

    Aall=[];
    resnorm_all = [];
    Iall = zeros(size(I));
    Ires = I-BkgrdInt;
    nopoints = length(indlist);
    
    ellipticity = length(indlist);
    
    for i = 1:nopoints

        waitbar(i/nopoints,waitb,['Fitting 2D Gaussians: ' num2str(i) '/' num2str(nopoints)]);

        index = indlist(i);

        Itemp=Ires;
        Itemp(LOC~=index)=0;

        [ytemp, xtemp] = find(Itemp~=0);
        xlo = min(xtemp);
        xhi = max(xtemp);
        ylo = min(ytemp);
        yhi = max(ytemp);
        
        Icut = Itemp(ylo:yhi,xlo:xhi);
        
        [mcut,ncut] = size(Icut);

        peakintensity = max(max(Icut(round(mcut/5):mcut-round(mcut/5),round(ncut/5):ncut-round(ncut/5))));%max(Icut(:));%Icut(peakpxy(index)-ylo,peakpxx(index)-xlo)
        mincellintensity = min(Icut(Icut>0));
        %Icut = Icut - mincellintensity;
        
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
        % Define lower and upper bounds [Amp,xo,wx,yo,wy,fi,offset]
        
%         lb = [peakintensity-2*sqrt(peakintensity),mcut/5,5,ncut/5,5,0,0];
%         ub = [peakintensity,mcut-mcut/5,minneighbourradius,ncut-ncut/5,minneighbourradius,pi/5,2*BkgrdInt];
        lb = [0,mcut/3,minneighbourradius/3,ncut/3,minneighbourradius/3,0,0];
        ub = [peakintensity,mcut-mcut/3,minneighbourradius*2,ncut-ncut/3,minneighbourradius*2,pi/4,peakintensity];
        
        if i == 1   % initial guess:
            %A0 = double([peakintensity,peakpxx(index)-xlo,10,peakpxy(index)-ylo,10,0,BkgrdInt]);
            A0 = double([peakintensity,peakpxx(index)-xlo,10,peakpxy(index)-ylo,10,0,mincellintensity]);
        else % use previous solution as initial guess:
            %A0 = double([peakintensity,peakpxx(index)-xlo,A(3),peakpxy(index)-ylo,A(5),0,A(7)]);
            A0 = double([peakintensity,peakpxx(index)-xlo,A(3),peakpxy(index)-ylo,A(3),0,mincellintensity]);
        end
        
        % Options for solver:
        %opts = optimset('Algorithm','levenberg-marquardt','Display','off','UseParallel',true); 
        opts = optimset('Display','off'); 

        % Fit sample data
        switch FitOrientation
            case 'dont' 
                %sig1=sig2
                [A,resnorm,res,flag,output] = lsqcurvefit(g1,A0(1:4),Xcut,Icut,lb(1:4),ub(1:4),opts);
                A(2) = A(2)+xlo-1;
                A(4) = A(4)+ylo-1;
                Iall = Iall + g1(A(1:4),X);
                Ires=Ires-g1(A(1:4),X);
                
                A(5) = A(3);    % sig1=sig2
                A(6) = 0;   % no angle fitted
                Aall = [Aall; A];
                resnorm_all = [resnorm_all resnorm];
            case 'dont2woffset' 
                [A,resnorm,res,flag,output] = lsqcurvefit(g2,[A0(1:4) A0(7)],Xcut,Icut,[lb(1:4) lb(7)],[ub(1:4) ub(7)],opts);
                A(2) = A(2)+xlo-1;
                A(4) = A(4)+ylo-1;
                Iall = Iall + g2([A(1:4) A(5)],X);
                Ires=Ires-g2([A(1:4) A(5)],X);
                
                %A(5) = A(3);    % sig1=sig2
                %A(6) = 0;   % no angle fitted
                Aall = [Aall; [A(1:4) A(3) 0 A(5)]];
                resnorm_all = [resnorm_all resnorm];
            case 'dont' 

            case 'fit1'
                [A,resnorm,res,flag,output] = lsqcurvefit(f,A0,Xcut,Icut,lb,ub,opts);
                A(2) = A(2)+xlo-1;
                A(4) = A(4)+ylo-1;
                Iall = Iall + f(A,X);
                Ires=Ires-f(A,X);
                Aall = [Aall; A];
                resnorm_all = [resnorm_all resnorm];
            otherwise, error('invalid entry');
        end
        %disp(output); % display summary of LSQ algorithm
        
      %if mod(i,50) == 0
      % if i<=5
        if rand>0.99
            
            
            figure;
            % Amplitude, X-Coord, X-Width, Y-Coord, Y-Width, Angle
            title(['Amplitude: ' num2str(A(1)) ' X: ' num2str(A(2)) ' Xw: ' num2str(A(3)) ' Y: ' num2str(A(4)) ' Yw: ' num2str(A(5)) ' alpha: ' num2str(rad2deg(A(6))) '?' ' fl: ' num2str((A(3)-A(5))/A(3))])

            subplot(1,2,1)
            imagesc(Icut)
            axis equal
            colorbar

            %figure
            subplot(1,2,2)
            %imagesc(g2([A(1),A(2)-xlo,A(3),A(4)-ylo,A(5)],Xcut))
            %imagesc(g1([A(1),A(2)-xlo,A(3),A(4)-ylo],Xcut))
            imagesc(f([A(1),A(2)-xlo+1,A(3),A(4)-ylo+1,A(5),A(6)],Xcut))
            axis equal
            colorbar
        end
        
        tempvar = [A(3) A(5)];  % used to easily find which of the two parameters is larger
        ellipticity(i) = (max(tempvar)-max(tempvar))/max(tempvar);  % definition of flattening: (a-b)/a, where a>b
        
    end
        
    % calculate MSE:
    MSE = ((peakpxx(indlist) - Aall(:,2)).^2 + (peakpxy(indlist) - Aall(:,4)).^2).^0.5;
    sortMSE = sort(MSE);
    index = find(MSE == MSE(end-1));
    Itemp = I;
    Itemp(LOC~=index)=0;

    [ytemp, xtemp] = find(Itemp~=0);
    xlo = min(xtemp);
    xhi = max(xtemp);
    ylo = min(ytemp);
    yhi = max(ytemp);
%     
%     figure;
%     imagesc(Itemp(ylo:yhi,xlo:xhi));
    
    figure;
    %scatter(Aall(:,2),Aall(:,4),[],ellipticity,'filled');
    scatter(Aall(:,2),Aall(:,4),[],resnorm_all,'filled')
    axis equal
    colormap

    Iall = Iall + BkgrdInt;
    delete(waitb);
    
    resnorm_all = resnorm_all';
    %Aall
end