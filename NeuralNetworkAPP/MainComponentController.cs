using DeepLearning_ImageClassification_Binary;
using Microsoft.ML;
using Microsoft.ML.Vision;
using Tensorflow.Keras.Engine;
using static Microsoft.ML.DataOperationsCatalog;

namespace NeuralNetworkAPP
{
    public class MainComponentController
    {
        private IDataView PreProcessedData { get; set; }
        public ITransformer TrainedModel { get; private set; }

        public MainComponentController(MLContext mlContext)
        {
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"ваш путь к assets"));
            var assetsRelativePath = Path.Combine(projectDirectory, "Assets");

            InitializePath(assetsRelativePath, mlContext);

            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: PreProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;

            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                //MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };

            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingPipeline.Fit(trainSet);

            

            //void SaveModel() => mlContext.Model.Save(trainedModel, trainSet.Schema, Path.Combine(workspaceRelativePath, "model.zip"));



            TrainedModel = trainedModel;
        }

        private void InitializePath(string assetsRelativePath, MLContext mlContext)
        {
            var images = NeuralNetwork.LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
            var imageData = mlContext.Data.LoadFromEnumerable(images);
            var shuffledData = mlContext.Data.ShuffleRows(imageData);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                       inputColumnName: "Label",
                       outputColumnName: "LabelAsKey")
                   .Append(mlContext.Transforms.LoadRawImageBytes(
                       outputColumnName: "Image",
                       imageFolder: assetsRelativePath,
                       inputColumnName: "ImagePath"));

            PreProcessedData = preprocessingPipeline
                                .Fit(shuffledData)
                                .Transform(shuffledData);
        }
    }
}
