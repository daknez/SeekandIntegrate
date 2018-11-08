function varargout = GUI(varargin)
% GUI MATLAB code for GUI.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI

% Last Modified by GUIDE v2.5 22-Jun-2017 14:51:32

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI is made visible.
function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI (see VARARGIN)

%add Toolbar for Zoom in / out and Pan 
figureToolBar = uimenu('Label','Zoom'); 
uimenu(figureToolBar,'Label','Zoom In','Callback','zoom on'); 
uimenu(figureToolBar,'Label','Zoom Out','Callback','zoom out'); 
uimenu(figureToolBar,'Label','Pan','Callback','pan on');

figureToolBar = uimenu('Label','Data'); 
uimenu(figureToolBar,'Label','Data Cursor','Callback','datacursormode on'); 

% Choose default command line output for GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUI wait for user response (see UIRESUME)
% uiwait(handles.guiMain);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in plotfilt.
function plotfilt_Callback(hObject, eventdata, handles)
% hObject    handle to plotfilt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in plotvoronoi.
function plotvoronoi_Callback(hObject, eventdata, handles)
% hObject    handle to plotvoronoi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in plotInt.
function plotInt_Callback(hObject, eventdata, handles)
% hObject    handle to plotInt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in segmentate.
function segmentate_Callback(hObject, eventdata, handles)
% hObject    handle to segmentate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global I;
global Ifilt;
global xyscale;
global ps;
global peakx;
global peaky;
global peakpxx;
global peakpxy;

binningfactor = str2num(get(handles.txtbinningf,'string'));
th = str2num(get(handles.lblThreshInt,'string'));

filterch = get(handles.chfilter,'string');
choice = filterch{get(handles.chfilter,'Value')};
if strcmp(choice,'none')
    Ifilt = I;
end
Ifilt_temp = imresize(Ifilt,1/binningfactor);

minpeak = str2num(get(handles.lblThreshInt,'string'))
Ifilt_temp(Ifilt_temp<minpeak)=0;

cmax = max(Ifilt(:));
cmin = min(Ifilt(:));
%-----------------------
% peak finding:
%-----------------------

minneighbourradius = round(str2num(get(handles.txtnearestn,'string')));
% ----------------------------------------------
mask = true(minneighbourradius); mask(round(minneighbourradius/2),round(minneighbourradius/2)) = 0;
peaks = Ifilt_temp > ordfilt2(Ifilt_temp,length(mask(mask==true)),mask);
[peakpxy,peakpxx] = find(peaks==1);
%-----------------------------------------------
% 
% [rr,cc] = meshgrid(1:size(Ifilt_temp,1),1:size(Ifilt_temp,2));
% count = 1;
% while count<10
%     Imax = max(Ifilt_temp(:))       
%     [psx,psy] = find(Ifilt_temp >= Imax-Imax*0.001,1);
%     mask = logical(sqrt((rr-psx).^2+(cc-psy).^2)<=minneighbourradius);
%     Ifilt_temp(mask) = 0;
%     peakpxx(count) = psx; % = [peakpxx; psx];
%     peakpxy(count) = psy; %[peakpxy; psy];
%     count=count+1
%     imagesc(Ifilt_temp);
% end
%-----------------------------------------------
% mit der Computer Vision toobox:
% hLocalMax = vision.LocalMaximaFinder;   % create the object
% hLocalMax.MaximumNumLocalMaxima = 1000;    % amount of maximal point
% hLocalMax.NeighborhoodSize = [minneighbourradius, minneighbourradius]; %�Neighborhood Size
% hLocalMax.Threshold = th;   %�Threshold
% 
% location = step(hLocalMax, Ifilt); %Apply the object to the Image I. location is an nx2�matrix  containing the coordinates of the maxima in the image � � � � � � � � � � � � � � � � � � � � � � � � � � � � � � � %matrix �containing the coordinates of the maxima in � � � � � � � � � � � � � � � � � � � � � � � � � � � � � � %the image 
% peakpxx = location(1,:);
% peakpxy = location(2,:);
%-----------------------------------------------

disp('peak finding done!');

peakpxy = peakpxy.*binningfactor;
peakpxx = peakpxx.*binningfactor;
peakx = double(peakpxx*ps-ps);
peaky = double(peakpxy*ps-ps);
disp(['number of peaks found ', num2str(length(peakx))]);

if length(peakx)>2
    [vx,vy] = voronoi(peakx,peaky);  % vx and vy contain the the finite vertices of the Voronoi edges
else
    msgbox('Less than 2 maxima found: no triangular mesh can be set up!');
    peakx
    peaky
end

ax = gca();
scatter(ax,peakx,peaky,'.')
xlim([0 max(xyscale)])
ylim([0 max(xyscale)])
axis square;
xlabel('Scale [nm]');
colormap jet;
cb = colorbar; 
ylabel(cb,'pixel intensity [counts]')
hold(ax, 'on');
imagesc(xyscale,xyscale,imresize(Ifilt_temp,binningfactor))
voronoi(peakx,peaky,'w')
caxis([cmin cmax])
hold(ax, 'off')

clear Ifilt_temp;
set(handles.integrate,'enable','on');
set(handles.txtmaxintrad,'enable','on');
set(handles.lblmaxintrad,'enable','on');
set(handles.cmdSaveMax,'enable','on');
set(handles.pushbutton21,'enable','on');

set(handles.lblHist,'enable','on');
set(handles.txtHistCat,'enable','on');


% --- Executes on button press in integrate.
function integrate_Callback(hObject, eventdata, handles)
% hObject    handle to integrate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
clear cellintensityimage
clear meancellintensities
clear integrationareas
clear cellintensities


global I;
global xyscale;
global peakx;
global peaky;
global peakpxx;
global peakpxy;
global cellintensityimage;
global cellintensities;
global filenames;
global filepath;
global maxfile;
global meancellintensities;
global integrationareas;

if get(handles.chbEvaluateAll,'value') == 0 || length(filenames) == 1

    N = min(size(I));
    maxcellradius = str2num(get(handles.txtmaxintrad,'string'));

    % Assign each pixel to a voronoi cell based on its distance to the peak positions
    % according to: https://de.mathworks.com/matlabcentral/newsreader/view_thread/303167
    % [X,Y] = meshgrid(xyscale,xyscale);
    % [SX,XX] = ndgrid(peakx, X);
    % D2X = (SX-XX).^2;
    % clear SX XX
    % [SY,YY] = ndgrid(peaky, Y);
    % D2Y = (SY-YY).^2;
    % clear SY YY
    % D2 = D2X+D2Y;
    % %D2 = (SX-XX).^2+(SY-YY).^2;
    % clear D2X D2Y
    % [~, LOC] = min(D2, [], 1);
    % LOC = reshape(LOC, size(X));


    disp('Assign pixels to cells...');
    [img_height, img_width] = size(I);
    clear ind
    if get(handles.chbIgnoreOutReg,'value')==1
        [img_height, img_width] = size(I);
        ind = (peakpxx<img_width-maxcellradius) & (peakpxx-maxcellradius>0) & (peakpxy<img_height-maxcellradius) & (peakpxy-maxcellradius>0);
        peakpxx = peakpxx(ind);
        peakpxy = peakpxy(ind);
        peakx = peakx(ind);
        peaky = peaky(ind);
    end
    LOC = GetNearestPeaks(img_height, img_width, peakpxx, peakpxy);
   
     disp('Determine Intensities...');

    % sum up all pixel intensities in each voroni cell: 

    cellintensities = zeros(1,max(LOC(:)));
    meancellintensities = zeros(1,max(LOC(:)));
    integrationareas =  zeros(1,max(LOC(:)));
    [rr,cc] = meshgrid(1:N);
    cellintensityimage = zeros(size(I));
   
    for i=1:max(LOC(:)) 
        cellmask = false(size(LOC));
        cellmask(LOC==i)=true;
        circlemask = logical(sqrt((rr-peakpxx(i)).^2+(cc-peakpxy(i)).^2)<=maxcellradius);
        cellmask = cellmask & circlemask;
        cellintensities(i) = sum(I(cellmask));
        cellintensityimage = cellintensityimage + cellmask.*cellintensities(i);
        integrationareas(i) = length(cellmask(cellmask==true));
        meancellintensities(i) = cellintensities(i)/length(cellmask(cellmask==true));
    end

    find(cellintensities==min(cellintensities(:)))

    ax = gca();
    scatter(ax,peakx,peaky,'.')
    xlim([0 max(xyscale)])
    ylim([0 max(xyscale)])
    axis square;
    cb = colorbar;
    ylabel(cb,'integrated intensity [counts]')
    xlabel('scale [nm]')
    colormap jet;
    hold(ax, 'on');
    imagesc(xyscale,xyscale,cellintensityimage);
    voronoi(peakx, peaky,'w')
    hold(ax, 'off');
    set(gcf,'PaperPositionMode','auto')
    print('temp','-dpng','-r0')

    set(handles.cmdSaveHist,'enable','on');
    set(handles.cmdPlotHist,'enable','on');
    set(handles.cmdPlotResult,'enable','on');
    
else  % evaluate all files in a folder (CheckBox Evaluate all files activated
   
    for file = filenames
        
        [~,~,ext] = fileparts(file{1});
        set(handles.txtpath,'string',file);
        
        if ext=='.tif'    
            % Read from Image (TIF) -File
            I = imread(strcat(filepath,file{1}));  

            prompt = {'Enter pixel size in nm'};
            pixelsize = inputdlg(prompt,'Enter pixel size',1,{'1'});
            ps = str2num(pixelsize{:});
            elseif ext=='.dm3'

            % Import from DM3-File:
            I_struct = DM3Import(strcat(filepath,file{1}));
            disp(I_struct)

            % Get size of image in pixels
            I = I_struct.image_data .* I_struct.intensity.scale; % Scale from counts to electrons   
            % Get pixel size (in nm)
            ps = I_struct.xaxis.scale;
            elseif ext=='.dm4'
                [I, ps, units] = ReadDMFile(strcat(filepath,file{1}));
                I = double(I');
              
            elseif ext=='.img'
                [I,t,dx,dy] = binread2D(strcat(filepath,file{1}));
                ps = dx;
               
            elseif ext=='.mat'
                load(strcat(filepath,file{1}),'I','ps');
            %     Itemp = zeros(512);
            %     Itemp(1:size(I,1), 1:size(I,2)) = I;
              %  I(~isfinite(I))=0;
            else
                msgbox('No File Selected');
                return;
        end

        N = min(size(I));
        xyscale = (0:N-1).*ps;

        cmax = max(I(:));
        cmin = min(I(:));
        Imax = max(I(:));

        Iorg = I; 
        Ifilt = I;
    
        minpeak = str2num(get(handles.lblThreshInt,'string'));
        
        ax = gca(); 
        imagesc(xyscale,xyscale,Ifilt);
        set(ax, 'YDir', 'normal')
        axis square
        xlabel('Scale [nm]');
        colormap gray;
        cb = colorbar; 
        caxis([cmin cmax])
        ylabel(cb,'counts')
        set(gcf,'PaperPositionMode','auto')
        print('temp','-dpng','-r0')
        
        filterch = get(handles.chfilter,'string');
        choice = filterch{get(handles.chfilter,'Value')};
        
        binningfactor = str2num(get(handles.txtbinningf,'string'));
        th = str2num(get(handles.lblThreshInt,'string'));

        if get(handles.chbSameMax,'value')==0
        
            switch choice
                case 'patchPCA'
                    a=get(handles.pumPCAw,'string')% string in it entirety
                    b=get(handles.pumPCAw,'value') % chosen value
                    startw = str2double(a(b,:)) % chosen string as value
                    %startw = str2num(get(handles.filterwidth,'string'));
                    PCn = str2num(get(handles.txtfiltercomp,'string'));
                    [Ifilt,~] = HirPatchPCA(I, startw, PCn, 1);

                    ax = gca(); 
                    imagesc(xyscale,xyscale,Ifilt);
                    set(ax, 'YDir', 'normal')
                    axis square
                    xlabel('Scale [nm]');
                    colormap gray;
                    cb = colorbar; 
                    caxis([minpeak cmax])
                    ylabel(cb,'counts')

                    set(handles.cmdPlotFiltered,'enable','on');
                    set(handles.cmdSaveFiltered,'enable','on');
                case 'none'
                    Ifilt = I;
                case 'mean'
                    w = str2num(get(handles.filterwidth,'string'));
                    h = ones(w,w) / w^2;
                    Ifilt = imfilter(I,h,'replicate');

                    ax = gca();
                    imagesc(xyscale,xyscale,Ifilt);
                    set(ax, 'YDir', 'normal')
                    axis square
                    xlabel('Scale [nm]');
                    colormap gray;
                    cb = colorbar; 
                    caxis([minpeak cmax])
                    ylabel(cb,'counts')

                    set(handles.cmdPlotFiltered,'enable','on');
                    set(handles.cmdSaveFiltered,'enable','on');
            end
        
            
            Ifilt_temp = imresize(Ifilt,1/binningfactor);
            Ifilt_temp(Ifilt_temp<minpeak)=0;

            minneighbourradius = round(str2num(get(handles.txtnearestn,'string')));
            % ----------------------------------------------
            mask = true(minneighbourradius); mask(round(minneighbourradius/2),round(minneighbourradius/2)) = 0;
            peaks = Ifilt_temp > ordfilt2(Ifilt_temp,length(mask(mask==true)),mask);
            [peakpxy,peakpxx] = find(peaks==1);

            disp('peak finding done!');

            peakpxy = peakpxy.*binningfactor;
            peakpxx = peakpxx.*binningfactor;
            peakx = (peakpxx*ps-ps);
            peaky = (peakpxy*ps-ps);
        end
        
        [vx,vy] = voronoi(peakx,peaky);  % vx and vy contain the the finite vertices of the Voronoi edges
        clear Ifilt_temp;
        
        disp('Assign pixels to cells...');
        maxcellradius = str2num(get(handles.txtmaxintrad,'string'));
        if get(handles.chbIgnoreOutReg,'value')==1
            [img_height, img_width] = size(I);
            ind = (peakpxx<img_width-maxcellradius) & (peakpxx-maxcellradius>0) & (peakpxy<img_height-maxcellradius) & (peakpxy-maxcellradius>0);
            peakpxx = peakpxx(ind);
            peakpxy = peakpxy(ind);
            peakx = peakx(ind);
            peaky = peaky(ind);
        end
        
        [img_height, img_width] = size(I);
        LOC = GetNearestPeaks(img_height, img_width, peakpxx, peakpxy);

        disp('Determine Intensities...');
        % sum up all pixel intensities in each voroni cell: 
        cellintensities = zeros(1,max(LOC(:)));
        [rr,cc] = meshgrid(1:N);
        cellintensityimage = zeros(size(I));
        meancellintensities = zeros(1,max(LOC(:)));
        integrationareas = zeros(1,max(LOC(:)));
        
        for i=1:max(LOC(:)) 
            cellmask = false(size(LOC));
            cellmask(LOC==i)=true;
            circlemask = logical(sqrt((rr-peakpxx(i)).^2+(cc-peakpxy(i)).^2)<=maxcellradius);
            cellmask = cellmask & circlemask;
            cellintensities(i) = sum(I(cellmask));
            cellintensityimage = cellintensityimage + cellmask.*cellintensities(i);
            integrationareas(i) = length(cellmask(cellmask==true));
            meancellintensities(i) = cellintensities(i)/length(cellmask(cellmask==true));
        end

        find(cellintensities==min(cellintensities(:)));
        
        ax = gca();
        scatter(ax,peakx,peaky,'.')
        xlim([0 max(xyscale)])
        ylim([0 max(xyscale)])
        axis square;
        cb = colorbar;
        ylabel(cb,'integrated intensity [counts]')
        xlabel('scale [nm]')
        colormap jet;
        hold(ax, 'on');
        himage = imagesc(xyscale,xyscale,cellintensityimage);
        %himage.AlphaData = 0.3;
        voronoi(peakx, peaky,'w')
        title('cells with corresponding intensities');
        %imagesc(xyscale,xyscale,Iorg);
        hold(ax, 'off');
        title('Integration result');

        index = (1:length(peakx))';
        Col_header = {'index','peak x coordinate [nm]','peak y coordinate [nm]','peak x coordinate [pixel]','peak y coordinate [pixel]','cell intensities [counts]', 'integration area [pixel]','mean cell intensity [counts/px]'};
        [~, temp, ~] = fileparts(file{1});
        if maxfile=='0' 
            filesave = [filepath temp];
        else
            [~, tempmax, ~] = fileparts(maxfile);
            filesave = [filepath temp '_' tempmax];
        end
        
        disp(['write ' filesave '_results.xls']);
        xlswrite([filesave '_results.xls'],[index, peakx, peaky, peakpxx, peakpxy, cellintensities' ,integrationareas', meancellintensities'],'Sheet1','A2'); % write data
        xlswrite([filesave '_results.xls'],Col_header,'Sheet1','A1');     %Write column header
        
        disp(['write ' filesave '_resultfig.png']);
        %savefig(gcf,[filepath file{1} '_resultfig.fig']);
        saveas(gcf,[filesave '_resultfig.png']);
        
    end
    msgbox('Evaluation finished.')
end

% --- Executes on button press in filter.
function filter_Callback(hObject, eventdata, handles)
% hObject    handle to filter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global I;
global xyscale;
global Ifilt;

binningfactor = str2num(get(handles.txtbinningf,'string'));

cmax = max(I(:));
cmin = min(I(:));
minpeak = str2num(get(handles.lblThreshInt,'string'));

filterch = get(handles.chfilter,'string');
choice = filterch{get(handles.chfilter,'Value')};
switch choice
    case 'patchPCA'
        a=get(handles.pumPCAw,'string')% string in it entirety
        b=get(handles.pumPCAw,'value') % chosen value
        startw = str2double(a(b,:)) % chosen string as value
        %startw = str2num(get(handles.filterwidth,'string'));
        PCn = str2num(get(handles.txtfiltercomp,'string'));
        [Ifilt,~] = HirPatchPCA(I, startw, PCn, 1);
        
        ax = gca(); 
        imagesc(xyscale,xyscale,Ifilt);
        set(ax, 'YDir', 'normal')
        axis square
        xlabel('Scale [nm]');
        colormap gray;
        cb = colorbar; 
        caxis([minpeak cmax])
        ylabel(cb,'counts')
        
        set(handles.cmdPlotFiltered,'enable','on');
        set(handles.cmdSaveFiltered,'enable','on');
    case 'none'
        Ifilt = I;
    case 'mean'
        w = str2num(get(handles.filterwidth,'string'));
        h = ones(w,w) / w^2;
        Ifilt = imfilter(I,h,'replicate');
        
        ax = gca();
        imagesc(xyscale,xyscale,Ifilt);
        set(ax, 'YDir', 'normal')
        axis square
        xlabel('Scale [nm]');
        colormap gray;
        cb = colorbar; 
        caxis([minpeak cmax])
        ylabel(cb,'counts')
        
        set(handles.cmdPlotFiltered,'enable','on');
        set(handles.cmdSaveFiltered,'enable','on');
end

% --- Executes on button press in Load.
function Load_Callback(hObject, eventdata, handles)
% hObject    handle to Load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global I;
global xyscale;
global ps;
global Iorg;
global Ifilt;
global filenames;
global filepath;
global maxfile;
global standardpath;

maxfile = '0';
if exist('standardpath')==0
    standardpath = pwd;
end

if get(handles.chbReadFolder,'value') == 0
    [file,filepath] = uigetfile({'*.dm3';'*.dm4';'*.img';'*.tif';'*.mat'},'Select file',standardpath);
else
    filepath = uigetdir(standardpath,'Select folder');
    filepath = [filepath '\'];
    fnames = [dir([filepath '\*.dm3']);dir([filepath '\*.img'])];
%   numfids = length(fnames);
%     vals = cell(1,numfids);
%     for K = 1:numfids
%         vals{K} = load(fnames(K).name);
%     end
    filenames = {fnames.name};%strcat(path,{fnames.name});
    file = [fnames(1).name];
end

if file ~= 0

    standardpath = filepath;
    [~,~,ext] = fileparts(file);

    if ext=='.tif'    
        % Read from Image (TIF) -File
        I = imread(strcat(path,file));  

        prompt = {'Enter pixel size in nm'};
        pixelsize = inputdlg(prompt,'Enter pixel size',1,{'1'});
        ps = str2num(pixelsize{:});
    elseif ext=='.dm3'

        % Import from DM3-File:
        I_struct = DM3Import(strcat(filepath,file));
        disp(I_struct)

        % Get size of image in pixels
        if exist('I_struct.intensity.scale')
            I = I_struct.image_data .* I_struct.intensity.scale; % Scale from counts to electrons   
        else
            I = I_struct.image_data;
        end
        % Get pixel size (in nm)
        ps = I_struct.xaxis.scale;
    elseif ext=='.dm4'
                [I, ps, units] = ReadDMFile(strcat(filepath,file));
                I = double(I');
    elseif ext=='.img'
        [I,t,dx,dy] = binread2D(strcat(filepath,file))
        if dx == dy
            ps = dx;
        else
    %         msgbox('pixels are not quadratic!','Warning!','warn');
    %         ps = (dx+dy)/2;
              ps = dx;
        end

    else
        msgbox('No File Selected');
        return;
    end

    N = min(size(I));
    xyscale = (0:N-1).*ps;

    cmax = max(I(:));
    cmin = min(I(:));
    Imax = max(I(:));

    Iorg = I;
    Ifilt = I;

    ax = gca(); 
    imagesc(xyscale,xyscale,Ifilt);
    set(ax, 'YDir', 'normal')
    axis square
    xlabel('Scale [nm]');
    colormap gray;
    cb = colorbar; 
    caxis([cmin cmax])
    ylabel(cb,'counts')

    set(handles.cmdPlotIorg,'enable','on');
    set(handles.segmentate,'enable','on');
    set(handles.sldThreshInt,'enable','on');
    set(handles.text3,'enable','on');
    set(handles.chfilter,'enable','on');
    set(handles.integrate,'enable','off');
    set(handles.txtmaxintrad,'enable','off');
    set(handles.lblmaxintrad,'enable','off');
    set(handles.cmdReset,'enable','off');
    set(handles.pumPCAw,'visible','off');
    set(handles.filterwidth,'visible','on');
    set(handles.cmdPlotFiltered,'enable','off');
    set(handles.cmdPlotResult,'enable','off');
    set(handles.cmdPlotHist,'enable','off');
    set(handles.cmdSaveFiltered,'enable','off');
    set(handles.cmdSaveMax,'enable','off');
    set(handles.cmdSaveHist,'enable','off');
    set(handles.cmdInvert,'enable','on');
    set(handles.cmdReadPeakCoords,'enable','on');

    set(handles.lblHist,'enable','off');
    set(handles.lblThrInt,'enable','on');
    set(handles.lblbinfac,'enable','on');
    set(handles.lblNearestN,'enable','on');
    set(handles.txtHistCat,'enable','off');
    if get(handles.chbReadFolder,'Value') == 0 
        set(handles.chbSameMax,'enable','off');
    else
        set(handles.chbSameMax,'enable','on');
    end

    set(handles.txtpath,'string',file);
    set(handles.sldThreshInt,'Max',Imax);
    set(handles.sldThreshInt,'Value',round(mean(I(:))));
    set(handles.chbEvaluateAll,'value',0)
    set(handles.lblThreshInt,'string',num2str(round(mean(I(:)))));
    set(handles.sldThreshInt, 'SliderStep', [1/Imax , 10/Imax ]);

end
% --- Executes on selection change in chfilter.
function chfilter_Callback(hObject, eventdata, handles)
% hObject    handle to chfilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns chfilter contents as cell array
%        contents{get(hObject,'Value')} returns selected item from chfilter

filterch = get(handles.chfilter,'string');
choice = filterch{get(handles.chfilter,'Value')};
switch choice
    case 'patchPCA'
        set(handles.text4,'enable','on');
        set(handles.filterwidth,'enable','on');
        set(handles.filtercomponents,'enable','on');
        set(handles.txtfiltercomp,'enable','on');
        set(handles.filter,'enable','on');
        
        set(handles.pumPCAw,'visible','on');
        set(handles.pumPCAw,'enable','on');
        set(handles.filterwidth,'visible','off');
    case 'none'
        set(handles.text4,'enable','off');
        set(handles.filterwidth,'enable','off');
        set(handles.filtercomponents,'enable','off');
        set(handles.txtfiltercomp,'enable','off');
        set(handles.filter,'enable','off');
        
        set(handles.pumPCAw,'visible','off');
        set(handles.filterwidth,'visible','on');
    case 'mean'
        set(handles.text4,'enable','on');
        set(handles.filterwidth,'enable','on');
        set(handles.filtercomponents,'enable','off');
        set(handles.txtfiltercomp,'enable','off');
        set(handles.filter,'enable','on');
        
        set(handles.pumPCAw,'visible','off');
        set(handles.filterwidth,'visible','on');
end

% --- Executes during object creation, after setting all properties.
function chfilter_CreateFcn(hObject, eventdata, handles)
% hObject    handle to chfilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function txtpath_Callback(hObject, eventdata, handles)
% hObject    handle to txtpath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtpath as text
%        str2double(get(hObject,'String')) returns contents of txtpath as a double


% --- Executes during object creation, after setting all properties.
function txtpath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtpath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function filterwidth_Callback(hObject, eventdata, handles)
% hObject    handle to filterwidth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of filterwidth as text
%        str2double(get(hObject,'String')) returns contents of filterwidth as a double


% --- Executes during object creation, after setting all properties.
function filterwidth_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filterwidth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function txtfiltercomp_Callback(hObject, eventdata, handles)
% hObject    handle to txtfiltercomp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtfiltercomp as text
%        str2double(get(hObject,'String')) returns contents of txtfiltercomp as a double


% --- Executes during object creation, after setting all properties.
function txtfiltercomp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtfiltercomp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function txtThreshInt_Callback(hObject, eventdata, handles)
% hObject    handle to txtThreshInt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtThreshInt as text
%        str2double(get(hObject,'String')) returns contents of txtThreshInt as a double


% --- Executes during object creation, after setting all properties.
function txtThreshInt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtThreshInt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function txtmaxintrad_Callback(hObject, eventdata, handles)
% hObject    handle to txtmaxintrad (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtmaxintrad as text
%        str2double(get(hObject,'String')) returns contents of txtmaxintrad as a double


% --- Executes during object creation, after setting all properties.
function txtmaxintrad_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtmaxintrad (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function sldThreshInt_Callback(hObject, eventdata, handles)
% hObject    handle to sldThreshInt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

global Ifilt
global xyscale

cmax = max(Ifilt(:));

set(handles.lblThreshInt,'string',round(get(handles.sldThreshInt,'value')));
set(handles.cmdReset,'enable','on');

minpeak = str2num(get(handles.lblThreshInt,'string'));

ax = gca(); 
imagesc(xyscale,xyscale,Ifilt);
set(ax, 'YDir', 'normal')
axis square
xlabel('Scale [nm]');
colormap gray;
cb = colorbar; 
caxis([minpeak cmax])
ylabel(cb,'counts')

% --- Executes during object creation, after setting all properties.
function sldThreshInt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sldThreshInt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function txtbinningf_Callback(hObject, eventdata, handles)
% hObject    handle to txtbinningf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtbinningf as text
%        str2double(get(hObject,'String')) returns contents of txtbinningf as a double


% --- Executes during object creation, after setting all properties.
function txtbinningf_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtbinningf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function txtnearestn_Callback(hObject, eventdata, handles)
% hObject    handle to txtnearestn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtnearestn as text
%        str2double(get(hObject,'String')) returns contents of txtnearestn as a double


% --- Executes during object creation, after setting all properties.
function txtnearestn_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtnearestn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function lblThreshInt_Callback(hObject, eventdata, handles)
% hObject    handle to lblThreshInt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of lblThreshInt as text
%        str2double(get(hObject,'String')) returns contents of lblThreshInt as a double



% --- Executes during object creation, after setting all properties.
function lblThreshInt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lblThreshInt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on key press with focus on lblThreshInt and none of its controls.
function lblThreshInt_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to lblThreshInt (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)

global Ifilt
global xyscale

set(handles.sldThreshInt,'value',round(str2num(get(handles.lblThreshInt,'string'))));
set(handles.cmdReset,'enable','on');

cmax = max(Ifilt(:));

set(handles.lblThreshInt,'string',round(get(handles.sldThreshInt,'value')));
set(handles.cmdReset,'enable','on');

minpeak = str2num(get(handles.lblThreshInt,'string'));

ax = gca(); 
imagesc(xyscale,xyscale,Ifilt);
set(ax, 'YDir', 'normal')
axis square
xlabel('Scale [nm]');
colormap gray;
cb = colorbar; 
caxis([minpeak cmax])
ylabel(cb,'counts')



% --- Executes on button press in cmdReset.
function cmdReset_Callback(hObject, eventdata, handles)
% hObject    handle to cmdReset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global I;

set(handles.sldThreshInt,'Value',round(mean(I(:))));
set(handles.lblThreshInt,'string',num2str(round(mean(I(:)))));
set(handles.cmdReset,'enable','off');


% --- Executes on selection change in pumPCAw.
function pumPCAw_Callback(hObject, eventdata, handles)
% hObject    handle to pumPCAw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pumPCAw contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pumPCAw


% --- Executes during object creation, after setting all properties.
function pumPCAw_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pumPCAw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in cmdPlotIorg.
function cmdPlotIorg_Callback(hObject, eventdata, handles)
% hObject    handle to cmdPlotIorg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global Iorg;
global xyscale;

cmax = max(Iorg(:));
cmin = min(Iorg(:));

figIorg = figure;
ax = gca();
set(figIorg,'DefaultAxesFontName', 'Arial', 'DefaultAxesFontSize', 14, 'DefaultAxesFontWeight', 'Demi' );
imagesc(xyscale,xyscale,Iorg);
imagesc(Iorg);
set(ax, 'YDir', 'normal')
axis square
xlabel('Scale [nm]');
colormap gray;
cb = colorbar; 
caxis([cmin cmax])
ylabel(cb,'counts')
imdisplayrange;
imcontrast(figIorg);
title('Original image');


% --- Executes on button press in cmdPlotFiltered.
function cmdPlotFiltered_Callback(hObject, eventdata, handles)
% hObject    handle to cmdPlotFiltered (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global Ifilt;
global xyscale;

cmax = max(Ifilt(:));
cmin = min(Ifilt(:));

figIfilt = figure;
ax = gca();
set(figIfilt,'DefaultAxesFontName', 'Arial', 'DefaultAxesFontSize', 14, 'DefaultAxesFontWeight', 'Demi' );
imagesc(xyscale,xyscale,Ifilt);
set(ax, 'YDir', 'normal')
axis square
xlabel('Scale [nm]');
colormap gray;
cb = colorbar; 
caxis([cmin cmax])
ylabel(cb,'counts')
imdisplayrange;
imcontrast(figIfilt);
title('Filtered image');

% --- Executes on button press in pushbutton17.
function pushbutton17_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in cmdPlotResult.
function cmdPlotResult_Callback(hObject, eventdata, handles)
% hObject    handle to cmdPlotResult (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global peakx;
global peaky;
global xyscale;
global cellintensityimage;

figResult = figure;
ax = gca();
scatter(ax,peakx,peaky,'.')
xlim([0 max(xyscale)])
ylim([0 max(xyscale)])
axis square;
cb = colorbar;
ylabel(cb,'integrated intensity [counts]')
xlabel('scale [nm]')
colormap jet;
hold(ax, 'on');
imagesc(xyscale,xyscale,cellintensityimage);
caxis([min(cellintensityimage(cellintensityimage~=0)) max(cellintensityimage(:))]);
voronoi(peakx, peaky,'w')
title('cells with corresponding intensities');
hold(ax, 'off');

title('Integration result');

% --- Executes on button press in cmdSaveFiltered.
function cmdSaveFiltered_Callback(hObject, eventdata, handles)
% hObject    handle to cmdSaveFiltered (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cmdSaveFiltered
global Ifilt;

[filename, pathname] = uiputfile({'*.tif';'*.*'},'Save as 32-bit data tif file',standardpath);
if filename ~= 0
    write32bittiff(Ifilt,[pathname, filename]);
end

% --- Executes on button press in cmdSaveMax.
function cmdSaveMax_Callback(hObject, eventdata, handles)
% hObject    handle to cmdSaveMax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cmdSaveMax
global peakx;
global peaky;
global peakpxx;
global peakpxy;
global cellintensities;
global filepath;
global meancellintensities;
global integrationareas;

standardpath = filepath;

[filename, pathname] = uiputfile({'*.xls';'*.csv';'*.*'},'Save as ...',standardpath);
if filename ~= 0
    [~,~,ext] = fileparts(filename);
    index = (1:length(peakx))';
    if ext == '.csv' 
        csvwrite([pathname, filename],[index, peakx, peaky, peakpxx, peakpxy, cellintensities']);
    elseif ext == '.xls'
        Col_header = {'index','peak x coordinate [pixel]','peak y coordinate [pixel]','peak x coordinate [nm]','peak y coordinate [nm]','cell intensities [counts]','integration area [pixel]','mean cell intensities [counts/pixel]'};
        xlswrite([pathname, filename],[index, peakx, peaky, peakpxx, peakpxy, cellintensities',integrationareas', meancellintensities'],'Sheet1','A2'); % write data
        xlswrite([pathname, filename],Col_header,'Sheet1','A1');     %Write column header
    end
end
% --- Executes on button press in cmdSaveHist.
function cmdSaveHist_Callback(hObject, eventdata, handles)
% hObject    handle to cmdSaveHist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cmdSaveHist
global cellintensities;

histogramvalues = str2num(get(handles.txtHistCat,'string'));
[counts,histcountsedges] = histcounts(cellintensities,histogramvalues);

[filename, pathname] = uiputfile({'*.csv';'*.*'},'Save as ...',standardpath);
if filename ~= 0
    csvwrite([pathname, filename],[histcountsedges(1:end-1)', counts']);
end

% --- Executes on button press in cmdPlotHist.
function cmdPlotHist_Callback(hObject, eventdata, handles)
% hObject    handle to cmdPlotHist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global cellintensities;

histogramvalues = str2num(get(handles.txtHistCat,'string'));
figcellintenshist = figure;
histogram(cellintensities,histogramvalues);
set(figcellintenshist,'DefaultAxesFontName', 'Arial', 'DefaultAxesFontSize', 16, 'DefaultAxesFontWeight', 'Demi' );
xlabel('integrated intensity [counts]')
ylabel('occurence')


function txtHistCat_Callback(hObject, eventdata, handles)
% hObject    handle to txtHistCat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtHistCat as text
%        str2double(get(hObject,'String')) returns contents of txtHistCat as a double


% --- Executes during object creation, after setting all properties.
function txtHistCat_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtHistCat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in cmdClose.
function cmdClose_Callback(hObject, eventdata, handles)
% hObject    handle to cmdClose (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cmdClose

%delete(handles.guiMain)


% --- Executes on button press in cmdReadPeakCoords.
function cmdReadPeakCoords_Callback(hObject, eventdata, handles)
% hObject    handle to cmdReadPeakCoords (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
clear peakpxx;
clear peakpxy;
clear peakx;
clear peaky;  

global peakpxx;
global peakpxy;
global ps;
global peakx;
global peaky;
global xyscale;
global Ifilt;
global standardpath;
global maxfile;

[maxfile, pathname] = uigetfile({'*.csv';'*.*'},'Load peak positions.',standardpath);
temp = csvread([pathname, maxfile]);

peakpxx = [];
peakpxy = [];
peakpxx = temp(:,1);
peakpxy = temp(:,2);

peakx = [];
peaky = [];
peakx = (peakpxx.*ps-ps);
peaky = (peakpxy.*ps-ps);

binningfactor = str2num(get(handles.txtbinningf,'string'));

[~,~] = voronoi(peakx,peaky);  % vx and vy contain the the finite vertices of the Voronoi edges

minpeak = str2num(get(handles.lblThreshInt,'string'));
cmax = max(Ifilt(:));

ax = gca();
scatter(ax,peakx,peaky,'.')
xlim([0 max(xyscale)])
ylim([0 max(xyscale)])
axis square;
xlabel('Scale [nm]');
colormap jet;
cb = colorbar; 
ylabel(cb,'pixel intensity [counts]')
hold(ax, 'on');
imagesc(xyscale,xyscale,imresize(Ifilt,binningfactor))
voronoi(peakx,peaky,'w')
caxis([minpeak cmax])
hold(ax, 'off')
set(gcf,'PaperPositionMode','auto')
print('temp','-dpng','-r0')
%print('-clipboard','-dmeta')


set(handles.integrate,'enable','on');
set(handles.txtmaxintrad,'enable','on');
set(handles.lblmaxintrad,'enable','on');
set(handles.cmdSaveMax,'enable','on');
set(handles.pushbutton21,'enable','on');

set(handles.lblHist,'enable','on');
set(handles.txtHistCat,'enable','on');

% --- Executes on button press in pushbutton21.
function pushbutton21_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global peakpxx;
global peakpxy;
global standardpath;

[filename, pathname] = uiputfile({'*.csv';'*.*'},'Save peak positions as ...',standardpath);
csvwrite([pathname, filename],[peakpxx, peakpxy]);


% --- Executes on button press in cmdInvert.
function cmdInvert_Callback(hObject, eventdata, handles)
% hObject    handle to cmdInvert (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global Ifilt
global xyscale

Ifilt = max(Ifilt(:))-Ifilt;

cmax = max(Ifilt(:));
cmin = min(Ifilt(:));

set(handles.sldThreshInt,'Max',cmax);
set(handles.sldThreshInt,'Min',cmin);

set(handles.sldThreshInt,'Value',round(mean(Ifilt(:))));
set(handles.lblThreshInt,'string',num2str(round(mean(Ifilt(:)))));

minpeak = str2num(get(handles.lblThreshInt,'string'));

ax = gca(); 
imagesc(xyscale,xyscale,Ifilt);
set(ax, 'YDir', 'normal')
axis square
xlabel('Scale [nm]');
colormap gray;
cb = colorbar; 
caxis([minpeak cmax])
ylabel(cb,'counts')


% --- Executes on button press in chbReadFolder.
function chbReadFolder_Callback(hObject, eventdata, handles)
% hObject    handle to chbReadFolder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chbReadFolder

if get(handles.chbReadFolder,'value') == 1 
    set(handles.chbEvaluateAll,'enable','on');
else
    set(handles.chbEvaluateAll,'enable','off');
    set(handles.chbSameMax,'enable','off');
end

% --- Executes on button press in chbEvaluateAll.
function chbEvaluateAll_Callback(hObject, eventdata, handles)
% hObject    handle to chbEvaluateAll (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chbEvaluateAll

if get(handles.chbEvaluateAll,'value') == 1
    set(handles.chbSameMax,'enable','on');
else
    set(handles.chbSameMax,'enable','off');
end

% --- Executes on key press with focus on sldThreshInt and none of its controls.
function sldThreshInt_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to sldThreshInt (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in chbSameMax.
function chbSameMax_Callback(hObject, eventdata, handles)
% hObject    handle to chbSameMax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chbSameMax


% --- Executes on button press in chbIgnoreOutReg.
function chbIgnoreOutReg_Callback(hObject, eventdata, handles)
% hObject    handle to chbIgnoreOutReg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chbIgnoreOutReg