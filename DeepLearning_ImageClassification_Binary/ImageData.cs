namespace DeepLearning_ImageClassification_Binary
{
    public class ImageData
    {
        public string? ImagePath { get; set; }

        public string? Label { get; set; }
    }
    public class ModelInput
    { 
        public byte[]? Image { get; set; }
        public UInt32 LabelAsKey { get; set; }
        public string? ImagePath { get; set; }
        public string? Label { get; set; }
    }
    public class ModelOutput
    {
        public string? ImagePath { get; set; }
        public string? Label { get; set; }
        public string? PredictedLabel { get; set; }
    }

}
