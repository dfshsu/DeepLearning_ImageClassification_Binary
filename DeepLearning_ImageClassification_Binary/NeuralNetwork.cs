using Microsoft.ML;

namespace DeepLearning_ImageClassification_Binary
{
    public class NeuralNetwork
    {
        public string? PredictedLabel { get; private set; }

        //Классификация одного изображения
        public void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
            ModelInput? image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).FirstOrDefault();
            ModelOutput? prediction = predictionEngine.Predict(image); OutputPrediction(prediction);

            OutputPrediction(prediction);
        }

        //Использование модели
        void OutputPrediction(ModelOutput prediction)
        {
            PredictedLabel = prediction.PredictedLabel;
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);

                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label[..index];
                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }
    }
}