% Question 3.1
C = 10;
tolerance = 0.00001;

[trD, trLb, valD, valLb, ~, ~] = HW4_Utils.getPosAndRandomNeg();

[w, bias] = train_svm(trD, trLb, C, tolerance);

HW4_Utils.genRsltFile(w, bias, "val", "result_file");
[ap_val, prec_val, rec_val] = HW4_Utils.cmpAP("result_file", "val");

% AP = 0.6192


% Question 3.2, 3.3

[trD, trLb, valD, valLb] = HW4_Utils.getPosAndRandomNeg();

% trD = [trD valD];
% trLb = [trLb; valLb];

PosD = trD(:, trLb > 0);
PosLb = trLb(trLb > 0);

NegD = trD(:, trLb < 0);
NegLb = trLb(trLb < 0);

C = 10;
tol = 0.0001;
overlap_threshold = 0.1;

apMat = zeros(10, 1);
objMat = zeros(10, 1);

load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, "train"), 'ubAnno');

[w, b, ~, alpha] = train_svm(trD, trLb, C, tol);

for i = 1 : 10 
    trD(:, (alpha < tol) & (trLb < 0)) = [];
    trLb((alpha < tol) & (trLb < 0), :) = [];
    
    newNegD = [];
    for j = 1 : 93
        im = sprintf('%s/trainIms/%04d.jpg', HW4_Utils.dataDir, j);
        im = imread(im);
        
        rect = HW4_Utils.detect(im, w, b);

        rectNeg = rect(:, rect(end, :)>0);
                
        [imH, imW, ~] = size(im);
        rectNeg = rectNeg(:, rectNeg(3, :) < imW);
        rectNeg = rectNeg(:, rectNeg(4, :) < imH);
        
        rectNeg = rectNeg(1:4, :);
        ubs = ubAnno{j};
        
        for k = 1 : size(ubs, 2)
            overlap = HW4_Utils.rectOverlap(rectNeg, ubs(:, k));                    
            rectNeg = rectNeg(:, overlap < overlap_threshold);
        end
        
        for k = 1 : size(rectNeg, 2)
            tmp = rectNeg(:, k);
            imReg = im(tmp(2):tmp(4), tmp(1):tmp(3),:);
            imReg = imresize(imReg, HW4_Utils.normImSz);
            
            feature = HW4_Utils.cmpFeat(rgb2gray(imReg));
            feature = HW4_Utils.l2Norm(feature);
            
            newNegD = [newNegD, feature];
        end
        
        % add maximum 1000 examples
        if size(newNegD, 2) > 1000
            break;
        end
    end
    
    trD = [trD, newNegD];
    trLb = [trLb; -1 * ones(size(newNegD, 2), 1)];
    
    [w, b, obj, alpha] = train_svm(double(trD), double(trLb), C, tol);
    
    HW4_Utils.genRsltFile(w, b, "val", "hard_mining_val");
    [ap] = HW4_Utils.cmpAP("hard_mining_val", "val");
    
    apMat(i) = ap;
    objMat(i) = obj;
end

disp("Objective Values:");
disp(objMat);

disp("Average Precision Values:");
disp(apMat);

iterations = [1 2 3 4 5 6 7 8 9 10];

figure
plot(iterations, objMat);
title('Objective Values Plot');
xlabel('Iteration');
ylabel('Objective Values');

figure
plot(iterations, apMat);
title('AP Plot');
xlabel('Iteration');
ylabel('APs');

%{
	Ub detection 92/92 (100.00%), elapse time:    45.8s
results have been saved to hard_mining_val
Objective Values:
  693.8269
  843.0114
  877.9326
  887.9181
  895.9787
  895.9787
  895.9787
  895.9787
  895.9787
  895.9787

Average Precision Values:
    0.8380
    0.8310
    0.8234
    0.8180
    0.8300
    0.8300
    0.8300
    0.8300
    0.8300
    0.8300
}%

% Question 3.4 and 3.5
HW4_Utils.genRsltFile(w, b, "test", "112671729");

%{
AP = 0.8071
}%
