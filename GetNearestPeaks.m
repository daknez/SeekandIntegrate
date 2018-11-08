function [ NearestPeaks ] = GetNearestPeaks( img_height, img_width, peakpxx, peakpxy )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

MaxRAMusage = 400; % in MBytes
BytesPerEntry = 8; % double precision
MaxSecRange = sqrt(MaxRAMusage * 1e6 / BytesPerEntry / length(peakpxx));
num_sectors = ceil( [img_height, img_width] ./ MaxSecRange );

secs_limits1 = [0, round( img_height/num_sectors(1) * (1:num_sectors(1)) )];
secs_limits2 = [0, round( img_width/num_sectors(2) * (1:num_sectors(2)) )];

% loop init
NearestPeaks = [];
for m = 1:numel(secs_limits1)-1
    RowOfPeaks = [];
    for n = 1:numel(secs_limits2)-1
        y = secs_limits1(m)+1 : secs_limits1(m+1);
        x = secs_limits2(n)+1 : secs_limits2(n+1);
        
        [X,Y] = meshgrid(x, y);

        [SX,XX] = ndgrid(peakpxx, X);
        [SY,YY] = ndgrid(peakpxy, Y);
        D2 = (SX-XX).^2 + (SY-YY).^2;
        [~, LOC] = min(D2, [], 1);
        LOC = reshape(LOC, secs_limits1(m+1)-secs_limits1(m), ...
            secs_limits2(n+1)-secs_limits2(n));
        RowOfPeaks = [RowOfPeaks, LOC];
    end
    NearestPeaks = [NearestPeaks; RowOfPeaks];
end

end

