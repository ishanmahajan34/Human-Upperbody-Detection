% question 2.1, 2,2, 2.3
function [w, bias, obj, alpha, support_vec] = train_svm(X, y, C, tolerance)

    [d, n] = size(X);
      
    % Initializing variables
    k = X' * X;
    H = diag(y) * k * diag(y);
    f = -1 * ones(n, 1);
    A = [];
    b = [];
    Aeq = y';
    beq = zeros(1, 1);
    lb = zeros(n, 1); 
    ub = C * ones(n, 1);

    % Calculating alpha
    [alpha, obj] = quadprog(H, f, A, b, Aeq, beq, lb, ub);
    obj = -obj;
    
    % Evaluating w and bias
    w = zeros(d, 1);
    for i = 1 : n
        w = w + ((alpha(i) * y(i)) * X(:, i));
    end

    bias = 0;
    for i = 1 : n
        if alpha(i) > tolerance && alpha(i) + tolerance < C
            bias = y(i) - (X(:, i)' * w);
        end
    end
    
    % Finding support vectors
    support_vec = size(find(alpha > tolerance), 1);
    
end


% question 2.4
C = 0.1;
tolerance = 0.00001;

load('/Users/ishan/Documents/CSE 512/hw4/q2_1_data.mat');

[w, bias, objective, alpha, support_vec] = train_svm(trD, trLb, C, tolerance);

val_pred = valD' * w + bias;
val_pred(val_pred >= 0) = 1;
val_pred(val_pred < 0) = -1;

accuracy_val = compute_accuracy(valLb, val_pred);
fprintf("Validation Data accuracy: [%f]\n", accuracy_val);
fprintf("Objective Value: [%f]\n", objective);
fprintf("Number of Support Vectors: [%d]\n", support_vec);
disp("Training Confusion Matrix - ");
disp(confusionmat(valLb, val_pred));


function [acc] = compute_accuracy(y_label, y_pred)
    acc = 100 * (sum(y_label == y_pred) / numel(y_label));
end

%{
Validation Data accuracy: [90.735695]
Objective Value: [24.764818]
Number of Support Vectors: [339]
Training Confusion Matrix - 
   152    32
     2   181

}%

% question 2.5
C = 10;
tolerance = 0.00001;

load('/Users/ishan/Documents/CSE 512/hw4/q2_1_data.mat');

[w, bias, objective, alpha, support_vec] = train_svm(trD, trLb, C, tolerance);

val_pred = valD' * w + bias;
val_pred(val_pred >= 0) = 1;
val_pred(val_pred < 0) = -1;

accuracy_val = compute_accuracy(valLb, val_pred);
fprintf("Validation Data accuracy: [%f]\n", accuracy_val);
fprintf("Objective Value: [%f]\n", objective);
fprintf("Number of Support Vectors: [%d]\n", support_vec);
disp("Training Confusion Matrix - ");
disp(confusionmat(valLb, val_pred));


function [acc] = compute_accuracy(y_label, y_pred)
    acc = 100 * (sum(y_label == y_pred) / numel(y_label));
end

%{
Validation Data accuracy: [97.820163]
Objective Value: [-112.146132]
Number of Support Vectors: [123]
Training Confusion Matrix - 
   180     4
     4   179
 }%


% question 2.6
% load data:

x_train = table2array(readtable('/Users/ishan/Documents/CSE 512/hw4/x_train.csv'));
y_train = table2array(readtable('/Users/ishan/Documents/CSE 512/hw4/y_train.csv', 'ReadVariableNames',false));

x_val = table2array(readtable('/Users/ishan/Documents/CSE 512/hw4/x_val.csv'));
y_val = table2array(readtable('/Users/ishan/Documents/CSE 512/hw4/y_val.csv', 'ReadVariableNames',false));

test_data = readtable('/Users/ishan/Documents/CSE 512/hw4/cse512hw4/Test_Features.csv');


% modify data:
X = [x_train x_val];
y = [y_train' y_val'];
y = y';

n = length(y);

y_train_1 = ones(n, 1);
y_train_2 = ones(n, 1);
y_train_3 = ones(n, 1);
y_train_4 = ones(n, 1);

for i = 1 : n
    if y(i) ~= 1
        y_train_1(i) = -1;
    end
    
    if y(i) ~= 2
        y_train_2(i) = -1;
    end
    
    if y(i) ~= 3
        y_train_3(i) = -1;
    end
    
    if y(i) ~= 4
        y_train_4(i) = -1;
    end
end



% one vs rest classifier
val1 = x_val' * w1 + bias1;
val2 = x_val' * w2 + bias2;
val3 = x_val' * w3 + bias3;
val4 = x_val' * w4 + bias4;

val_1_vs_all = [val1 val2 val3 val4];

[~, argmax] = max(val_1_vs_all');
count = 0;
for i = 1:2000
    if argmax(i) == y_val(i)
        count = count + 1;
    end
end

accuracy_1_vs_all = count / 2000;

% predicting on test data
Id = test_data.Var1;
x_test = table2array(test_data(:, 2:513));

C = 0.00057;
tolerance = 0.00001;
[w1, bias1, objective1, alpha1] = train_svm(X, y_train_1, C, tolerance);
disp("1st done")
[w2, bias2, objective2, alpha2] = train_svm(X, y_train_2, C, tolerance);
disp("2nd done")
[w3, bias3, objective3, alpha3] = train_svm(X, y_train_3, C, tolerance);
disp("3rd done")
[w4, bias4, objective4, alpha4] = train_svm(X, y_train_4, C, tolerance);
disp("4th done")

val1 = x_test * w1 + bias1;
val2 = x_test * w2 + bias2;
val3 = x_test * w3 + bias3;
val4 = x_test * w4 + bias4;

val_1_vs_all = [val1 val2 val3 val4];

[~, Category] = max(val_1_vs_all');

Category = Category';

output_table = table(Id, Category);


writetable(output_table, "/Users/ishan/Desktop/predTestLabels.csv");


% Kaggle Score = 0.50666