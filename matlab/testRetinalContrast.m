% This script tests the retinal contrast function on scene luminances.
% First the parameters are set. Then the function computes input scene 
% luminances by combining a 2D input map of the scene and a table of 
% telephotometer readings of the scene. This is done because telephotometers 
% are the only reliable method of estimating accurate luminances without 
% being affected by the effects of glare. 
%
% mapInput (unit8): 2D paint-by-numbers map of scene used to calculate luminance 
% conversionTable (double): calibration measurements of scene luminances




clear all
close all

tic 

%--------------------------------------------------------setting parameters

parameters.age=25; %age of the observer
parameters.pigmentationFactor = 0.5; %(p=0 for very dark eyes, p=0.5 for brown eyes, p=1.0 for blue-green caucasians, up to p=1.2 for blue eyes).
parameters.pixelSize = 0.1664; %in mm
parameters.viewingDistance =  360; %in mm
parameters.verbose= true; % display detailed messages during runtime
parameters.range=5.4; % output range in log units


%----------------------------------------------------------Loading map file

inputMapFilename='../data/scene1/map.tiff'; %inputMapFilename of the image to be processed
mapInput = imread(inputMapFilename);%load image file
if (size(mapInput,3)>1)
    mapInput=rgb2gray(mapInput);
end
% mapInput=imresize(mapInput,0.5);

%display original image
figure, imshow(mapInput);
colorbar
title('INPUT map');

conversionTable = load('../data/scene1/LUT.txt');
% q=xlsread('sTable.xls');
% conversionTable=q(:,2);



%defining visualization map for pseudocolors
pseudocolors=[
0	0	0
0.0825	0.061875	0
0.165	0.12375	0
0.2475	0.1134375	0
0.33	0.103125	0
0.4125	0.07734375	0
0.495	0.0515625	0
0.5775	0.19078125	0
0.66	0.33	0
0.7012	0.28875	0
0.7424	0.2475	0
0.7836	0.20625	0
0.8248	0.165	0
0.866	0.12375	0
0.9072	0.0825	0
0.9484	0.04125	0
1	0	0
1	0	0.125
1	0	0.25
1	0	0.375
1	0	0.5
1	0	0.625
1	0	0.75
1	0	0.875
1	0	1
0.91625	0.04125	1
0.8325	0.0825	1
0.74875	0.12375	1
0.665	0.165	1
0.58125	0.20625	1
0.4975	0.2475	1
0.41375	0.28875	1
0.33	0.33	1
0.35125	0.41375	1
0.3725	0.4975	1
0.39375	0.58125	1
0.415	0.665	1
0.43625	0.74875	1
0.4575	0.8325	1
0.47875	0.91625	1
0.5	1	1
0.5	1	0.9375
0.5	1	0.875
0.5	1	0.8125
0.5	1	0.75
0.5	1	0.6875
0.5	1	0.625
0.5	1	0.5625
0.5	1	0.5
0.5625	1	0.4375
0.625	1	0.375
0.6875	1	0.3125
0.75	1	0.25
0.8125	1	0.19
0.875	1	0.13
0.925	1	0.06
1	1	0
1	1	0.125
1	1	0.25
1	1	0.375
1	1	0.5
1	1	0.625
1	1	0.75
1	1	0.88
1	1	1];





%----------------------------------------------Calculating scene luminances

%applying the conversion table on the map scene

conversionTable=10.^conversionTable;
% conversionTable(1)=0;%first digit should be 0 to represent opaque areas
sceneLuminance=conversionTable(mapInput+1);%taking care of Matlab's indexing [0,255]->[1,256]


%statistics of the scene luminance
maxSceneLuminance = max(max(sceneLuminance));
minSceneLuminance = min(min(sceneLuminance));
rangeSceneLuminance = maxSceneLuminance/minSceneLuminance;
meanSceneLuminance = mean(mean(sceneLuminance));
fprintf('\n Scene luminance statistics');
fprintf(['\n max=' num2str(maxSceneLuminance)]);
fprintf(['\n min=' num2str(minSceneLuminance)]);
fprintf(['\n mean=' num2str(meanSceneLuminance)]);
fprintf(['\n range=' num2str(rangeSceneLuminance) '\n']);


sceneLuminance=(sceneLuminance./maxSceneLuminance);%normalizing scene luminance to the maximum
sceneLuminanceLog=log10(sceneLuminance); %range=[-100,0]
sceneLuminanceLogRange=sceneLuminanceLog;
sceneLuminanceLogRange(sceneLuminanceLogRange<-parameters.range)=-parameters.range;%truncate anything below the output log range [-logrange,0]
sceneLuminanceLogRange=sceneLuminanceLogRange+parameters.range; %[0,logRange]
sceneLuminanceLogRange=sceneLuminanceLogRange./parameters.range; %[0,1]
sceneLuminanceLogRange=sceneLuminanceLogRange.*255;%[0,255]

imwrite(uint8(sceneLuminanceLogRange),'sceneLuminanceLogRange.tiff','tiff');


%display original image
figure, imshow(uint8(sceneLuminanceLogRange));
colorbar
title('INPUT Scene Log Luminance Range');
colormap(pseudocolors);
print -painters -dpng -r300 sceneLuminanceLogRange.png

%----------------------------------------------------------estimating glare

%calling the glare estimation function 'computeRetinalContrast.m'
[retinalContrast]=computeRetinalContrast(sceneLuminance, parameters);


%-----------------------------------displaying and writing the glare images

%writing the logarithmic glare image with and without pseudocolors

retinalContrastLog=log10(retinalContrast);
retinalContrastLogRange=retinalContrastLog;
retinalContrastLogRange(retinalContrastLogRange<-parameters.range)=-parameters.range;%[-logrange,0]
retinalContrastLogRange=retinalContrastLogRange+parameters.range;%[0,logrange]
retinalContrastLogRange=retinalContrastLogRange./parameters.range;%[0,1]
retinalContrastLogRange=retinalContrastLogRange.*255;%[0,255]
retinalContrastLogRange=uint8(retinalContrastLogRange);



imwrite(retinalContrastLogRange,'retinalContrastLogRange.tiff','tiff');

figure, imshow(retinalContrastLogRange);
colorbar
colormap(pseudocolors);
title(['Retinal Contrast Log Range = ' num2str(parameters.range)]);
print -painters -dpng -r300 retinalContrastLogRange.png



%SCALE PSEUDOCOLOR TO OUTPUT RANGE (scene dependent)
%Glare reduces the range of output. The following code reduces the pseudocolor range to that of the output. 

minRetinalContrastLogRangeOut = min(min(retinalContrastLog));
maxRetinalContrastLogRangeOut = max(max(retinalContrastLog));

rangeRetinalContrastLogRangeOut = maxRetinalContrastLogRangeOut-minRetinalContrastLogRangeOut;

retinalContrastLogRangeOut=retinalContrastLog;
retinalContrastLogRangeOut= 255.*((retinalContrastLogRangeOut-minRetinalContrastLogRangeOut)./rangeRetinalContrastLogRangeOut);%rescale to output
retinalContrastLogRangeOut=uint8(retinalContrastLogRangeOut);

figure, imshow(retinalContrastLogRangeOut);
colorbar
colormap(pseudocolors);
title(['Retinal Contrast Log Range Output = ' num2str(rangeRetinalContrastLogRangeOut)]);
print -painters -dpng -r300 retinalContrastLogRangeOut.png

toc



