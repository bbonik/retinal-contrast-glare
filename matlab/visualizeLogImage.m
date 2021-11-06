function [ imageLogDisplay ] = visualizeLogImage( imageLog, logRange )

%Preprocessing for displaying a logarithmic image
%
%--------------------------------------------------------------------------
% INPUTS
%
% imageLog (double): logarithmic (log10) encoding of a [0,1] image. Since
%                    the original image is in the interval [0,1], its
%                    logarithmic encoding (imageLog) is in the interval
%                    (-inf,0].
%                    
% logRange (double): range (in log units) that will be applied on the 
%                    visualization output 
%
%--------------------------------------------------------------------------
% OUTPUTS
%
% imageLogDisplay (uint8): visualization output in the interval [0,255] of 
%                          the logarithmic input image 
%


imageLogDisplay=imageLog;
imageLogDisplay(imageLogDisplay<-logRange)=-logRange;%truncate anything below the output log range [-logRange,0]
imageLogDisplay=imageLogDisplay+logRange; %[0,logRange]
imageLogDisplay=imageLogDisplay./logRange; %[0,1]
imageLogDisplay=uint8(imageLogDisplay.*255);%[0,255]





end

