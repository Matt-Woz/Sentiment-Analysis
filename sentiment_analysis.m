%Importing, converting and cleaning data
tStart = tic;
table = readtable('text_emotion_data.csv');
string = table.Content;
%string = eraseURLs(string); %Discussed in document
%string = erasePunctuation(string); %Discussed in document
documents = tokenizedDocument(string);
bag = bagOfWords(documents);
newBag = removeWords(bag,stopWords);
count = 100;
newerBag = removeInfrequentWords(newBag,count);
%Create TF-IDF
M1 = tfidf(newerBag);
M1 = full(M1);
%Extracting sentiment column
str = table.sentiment;
%Select Training data
labelTraining = str(1:6921);
M1Training = M1(1:6921,:);
%Select Testing data
labelTesting = str(6922:8651);
M1Testing = M1(6922:8651,:);
%Run model and calculate accuracy
pred = getPredict(M1Training,M1Testing,labelTraining,@fitcnb); %Change 4th parameter for different algorithm
accuracy = getAccuracy(pred,labelTesting);
%Create confusion chart + print the accuracy
confusionchart(labelTesting, pred)
disp(accuracy)
tEnd = toc(tStart)
%Function to run model with algorithm as variable
function predictions = getPredict(trainingFeatures, testingFeatures,label,algorithm)
model = algorithm(trainingFeatures, label);
predictions = predict(model,testingFeatures);
end
%Function to calculate accuracy of model
function accuracy = getAccuracy(prediction, testingLabel)
correctPredictions = 0;
for i=1:length(prediction)
    if isequal(prediction(i),testingLabel(i))
        correctPredictions = correctPredictions + 1;
    end
end
accuracy = correctPredictions / length(prediction);
end