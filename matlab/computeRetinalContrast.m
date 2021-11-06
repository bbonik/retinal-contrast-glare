function [retinalContrast]= computeRetinalContrast(sceneLuminance, parameters)

% RetinalContrast:
% Calculation of the array of light falling on the retina, based on the
% equations of Vos&van den Berg (1999) CIE standard.
%
% The function uses the glare spread function in order to estimate the
% retinal image derived by the particular luminace inputs. It requires the 
% use of the Image Processing toolbox. Attention: large input scenes may 
% need a lot of time to be processed!
%
%--------------------------------------------------------------------------
% INPUTS
%
% parameters: structure of the set of model parameters defined in
%             Vos&van den Berg (1999) CIE standard
% sceneLuminance (double): linear calibrated scene luminance array 
%                          normalized in the range [0,1]
%
%--------------------------------------------------------------------------
% OUTPUTS
%
% retinalContrast (double): linear retinal contrast array 

 


%-------------------------------------------------Calculating filter kernel
% According to equation (8) of  Vos&van den Berg (1999) CIE standard
%
% The glare spread function from the paper is used in order to create a 2D
% convolution kernel. After that, the kernel is convolved with the
% input luminance image in order to estimate the cummulated contributions
% of different points on the scene, to the retinal image.


fprintf('2. Calculating filterKernel...\n');

radius = max(size(sceneLuminance));
filterKernel = double(zeros((2*radius) + 1));


for i = 1:(2*radius + 1)
    for j = 1:(2*radius + 1)
        
        dist = parameters.pixelSize*sqrt((i - (radius + 1))^2 + (j - (radius + 1))^2);
        th = atand(dist/parameters.viewingDistance); %glare angle theta
        
        filterKernel(i,j) = (1 - 0.008*(parameters.age/70)^4) * ...
            (9.2e6/(1 + (th/0.0046)^2).^1.5 + ...
            1.5e5/(1 + (th/0.045)^2).^1.5) + ...
            (1 + 1.6*(parameters.age/70)^4) * ...
            ((400/(1 + (th/0.1).^2) + 3e-8*th^2) + ...
            parameters.pigmentationFactor*(1300/(1 + (th/0.1)^2)^1.5 + ...
            0.8/(1 + (th/0.1)^2)^0.5)) + ...
            2.5e-3*parameters.pigmentationFactor;
        
        filterKernel(i,j)=filterKernel(i,j)*cosd(th);% correction for flat target instead of sphere
        
    end
end


%----------------------------------------Normalization of the filter kernel
% The sum of all elements of the filter kernel should sum up to 1, in order
% not to add any DC constant during convolution.


fprintf('3. Correct sum of all pixels to 1.0...\n');


filterKernel=filterKernel./sum(filterKernel(:));


%-------------------------------------------Performing the actual filtering

fprintf('4. Filtering (this may take time for large maps)...\n');


% Convolution using imfilter function (requires Image Processing toolbox).
% This function minimizes the impact of boudnary conditions by replicating
% the border values of the input luminance image. It is also FFT-based, so
% it is faster compared to a typical convolution.


retinalContrast=imfilter(sceneLuminance,filterKernel,'replicate');


%statistics of the retinal contrast image
maxRetinalContrast = max(max(retinalContrast));
minRetinalContrast = min(min(retinalContrast));
rangeRetinalContrast = maxRetinalContrast/minRetinalContrast;
meanRetinalContrast = mean(mean(retinalContrast));
fprintf('\n Retinal contrast statistics\n');
fprintf(['max=' num2str(maxRetinalContrast)]);
fprintf(['\n min=' num2str(minRetinalContrast)]);
fprintf(['\n mean=' num2str(meanRetinalContrast)]);
fprintf(['\n range=' num2str(rangeRetinalContrast)]);




