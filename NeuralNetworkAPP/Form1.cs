using DeepLearning_ImageClassification_Binary;
using Microsoft.ML;

namespace NeuralNetworkAPP
{
    public partial class Form1 : Form
    {
        private string? TestImagePath { get; set; }
        private NeuralNetwork Network { get; set; }
        private MLContext MlContext { get; set; }

        public MainComponentController ComponentController { get; set; }

        public Form1()
        {
            InitializeComponent();
            Network = new NeuralNetwork();
            MlContext = new MLContext();
            ComponentController = new MainComponentController(MlContext);
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void ImageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                TestImagePath = openFileDialog1.FileName;
                IDataView PreProcessedData = LoadNeuralNetworkData();

                Network.ClassifySingleImage(MlContext, PreProcessedData, ComponentController.TrainedModel);

                LoadData();
            }

            IEnumerable<ImageData> LoadImageData()
            {

                yield return new ImageData
                {
                    ImagePath = TestImagePath,
                    Label = ""
                };

            }

            IDataView LoadNeuralNetworkData()
            {
                var image = LoadImageData();

                var imageData = MlContext.Data.LoadFromEnumerable(image);

                var preprocessingPipeline = MlContext.Transforms.Conversion.MapValueToKey(
                       inputColumnName: "Label",
                       outputColumnName: "LabelAsKey")
                   .Append(MlContext.Transforms.LoadRawImageBytes(
                       outputColumnName: "Image",
                       imageFolder: "",
                       inputColumnName: "ImagePath"));

                var PreProcessedData = preprocessingPipeline
                                    .Fit(imageData)
                                    .Transform(imageData);
                return PreProcessedData;
            }
        }
        private void LoadData()
        {
            listView1.Items.Clear();

            ImageList imageList = new()
            {
                ImageSize = new Size(320, 320)
            };

            imageList.Images.Add(new Bitmap(TestImagePath));

            Bitmap emptyImage = new(320, 320);

            using (Graphics gr = Graphics.FromImage(emptyImage))
            {
                gr.Clear(Color.White);
            }

            imageList.Images.Add(emptyImage);

            listView1.SmallImageList = imageList;
            AddItemInListView();
        }

        private void AddItemInListView()
        {
            string lable = "";
            for (int i = 0; i < 1; i++)
            {
                if (Network.PredictedLabel == "UD")
                {
                    lable += "Трещин на изображении не найдено";
                }
                else
                {
                    lable += "Трещины на изображении найдены";
                }
                ListViewItem listViewItem = new(new string[] { "", lable })
                {
                    ImageIndex = i
                };

                listView1.Items.Add(listViewItem);
            }
        }

        private void ExitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            MessageBoxButtons msb = MessageBoxButtons.YesNo;
            String message = "Вы действительно хотите выйти?";
            String caption = "Выход";
            if (MessageBox.Show(message, caption, msb) == DialogResult.Yes)
                this.Close();
        }
    }
}