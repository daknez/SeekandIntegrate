function [Output, Residual] = patchpca(I, startPCn,endPCn,w)
 
% ----------------------------------------
% random patches:
%-----------------------------------------

% M = 1000;       % number of patches
% w = 64;         % size of single patch
% Y = zeros(64,64,1000);
% sig = zeros(64,64);
% [co,ro] = size(I);
% 
% for i=1:M
%     
%     randompx = round(rand()*co);
%     if (randompx < w)
%         randompx = randompx + w;
%     elseif (randompx > co-w)
%         randompx = randompx - w;
%     end
%         
%     randompy = round(rand()*ro);
%     if (randompy < w)
%         randompy = randompy + w;
%     elseif (randompy > co-w)
%         randompy = randompy - w;
%     end
%     
%     Y(:,:,i) = I(randompx:randompx+w-1,randompy:randompy+w-1);
% 
% end
% 
% Ymn = mean(Y,3);
% 
% % data = Y - repmat(Ymn,[1 1 M]);
% 
% for i=1:M 
%     sig = sig+(Y(:,:,i)*Y(:,:,i)');
% end
%     
% sig = 1/M.*sig -Ymn*Ymn';     % covariance matrix
% 
% [U,S,V] = svd(sig);
% 
% fig2 = figure;
% variances = diag(S) .* diag(S); % compute variances (Eigenvalues)
% plot(variances/sum(variances),'*');  % scree plot of variances
% set(gca,'YScale','log')
% xlim([0 w]);
% xlabel('eigenvector number');
% ylabel('explained variance ratio');
% 
% Veff = V(:,1:1); % select range of PCs to keep
% 
% Vdata = Veff' * sig;
% PCA_Out = Veff * Vdata; % convert back to original basis
% 
% figure;
% imagesc(PCA_Out);
%------------------------------------------------------------------------

%------------------------------------
% patches without overlapp:
%------------------------------------
%w = 16; % width of each patch
[co,ro] = size(I);
PCn = endPCn;

Output = zeros(size(I));
Residual = zeros(size(I));

paco = (co / w); % no of patch columns
N = (co / w)^2; % number of patches necessary (number of trials)
M = w^2; % number of pixels in a single patch (number of dimensions)
if co ~= ro || mod(co,w)~=0   % error handling
    disp('ERROR: PCA-Filtering failed due to non square image or wron patch width');
end

data = zeros(M,N);

% create data set width M vectors each with dimension = number of pixels in a patch
D = mat2cell(I,w*ones(1,paco),w*ones(1,paco));
Res = cell(size(D));
Out = cell(size(D));

for i=1:N
    %data(i,:) = D{i}(:)-mean(D{i}(:));  % with patch-wise subtraction of mean
    data(:,i) = D{i}(:);
end

dimmn = mean(data,2);
%data = data - repmat(mean(data,2),1,N);
data = bsxfun(@minus, data, dimmn); % shift data to zero-mean

Z = 1/sqrt(N-1) * data';

%covZ = Z' * Z; % covariance matrix of Z (Die Varianz ist die Kovarianz einer Zufallsvariablen mit sich selbst)
%[U,S,V] = svd(covZ); % Singular value decomposition

[U,S,PC] = svd(Z,'econ'); % Singular value decomposition (ohne covarianzmatrix nach Shlens.2005)

% columns of V are the principal component directions, the SVD automatically sorts these components in decreasing order of "principality"
%variances = diag(S) .* diag(S); % compute variances (Eigenvalues)
variances = diag(S'*S)' / (size(I,1)-1); %variance explained

% without using SVD (Shlens.2005) :
% covZ = Z' * Z;
% [V,variances] = eig(covZ);
% variances = diag(variances);
% 
% [junk, rindices] = sort(-1*variances);
% variances = variances(rindices);
% V = V(:,rindices);

var_normalised = variances/sum(variances);
% fig2 = figure;
% plot(var_normalised,'*');  % scree plot of variances
% set(gca,'YScale','log')
% xlim([0 50]);
% xlabel('eigenvector number');
% ylabel('explained variance ratio');

% automatically select number of components:
if PCn == 0
    PCn = length(var_normalised(var_normalised > 2e-2));
end

%title(['scree plot (patchsize ' num2str(w) '), ' num2str(PCn) ' components selected']);

PCeff = PC(:,startPCn:PCn); % select range of PCs to keep

PCres = PC(:,PCn:end); % residual PCs

PCdata = PCeff' * data; % project data onto PCs

PCA_res = PCres * (PCres' * data);
%ratio = co / (2 * PCn + 1) % compression ratio
PCA_Out = PCeff * PCdata; % convert back to original basis

PCA_Out = bsxfun(@plus, PCA_Out, dimmn); % shift data back
PCA_res = bsxfun(@plus, PCA_res, dimmn); % shift data back

for i=1:N
    Out{i} = reshape(PCA_Out(:,i),w,w);%+mean(D{i}(:));
    Res{i} = reshape(PCA_res(:,i),w,w);%+mean(D{i}(:));
end
Output = cell2mat(Out);
Residual = cell2mat(Res);

