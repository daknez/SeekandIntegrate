function [I,ps] = ReadMicImageFromFile(standardpath)

[file,path] = uigetfile({'*.dm3';'*.img';'*.tif';'*.mat'},'Select imagefile for deconvolution',standardpath);
[pathstr,filename,ext] = fileparts(file);

if ext=='.tif'    
    
    % Read from Image (TIF) -File
    I = imread(strcat(path,file));  
    N = min(size(I));
    
    prompt = {'Enter field of View in nm'};
    fieldofview = inputdlg(prompt,'Enter field of view',1,{'5.7'});
    ps = str2num(fieldofview{:})./N;
    xyscale = (0:N-1).*ps;
elseif ext=='.dm3'

    % Import from DM3-File:
    I_struct = DM3Import(strcat(path,file));
    disp(I_struct)
    
    % Get size of image in pixels
    I = I_struct.image_data .* I_struct.intensity.scale; % Scale from counts to electrons   
    % Get pixel size (in nm)
    ps = I_struct.xaxis.scale;
    
%     fig1 = figure; movegui('north');
%     imagesc(xyscale,xyscale,I);
%     set(fig1,'DefaultAxesFontName', 'Arial', 'DefaultAxesFontSize', 14, 'DefaultAxesFontWeight', 'Demi' );
%     axis equal
%     if exist('I_struct.image_name') ~= 0
%          title({I_struct.image_name strcat('(Magnification: ',num2str(I_struct.mag),'x)')});
%     end
%     xlabel('Scale [nm]');
%     colormap gray;
%     cb = colorbar; 
%     caxis([cmin cmax])
%     ylabel(cb,'counts')

elseif ext=='.img'
    [I,t,dx,dy] = binread2D(strcat(path,file));
    if dx == dy
        ps = dx;
    else
        msgbox('pixels are not quadratic!','Warning!','warn');
        ps = (dx+dy)/4;
    end
elseif ext=='.mat'
    load(strcat(path,file),'I','ps');
    Itemp = zeros(512);
    Itemp(1:size(I,1), 1:size(I,2)) = I;
    
    I(~isfinite(I))=0;
else
    msgbox('No File Selected');
    return;
end
end