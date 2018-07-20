using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Linq;
using System.Text.RegularExpressions;

namespace GettingStarted
{
    class Program
    {
        public class IrisData
        {
            [Column("0")]
            public float SepalLength;

            [Column("1")]
            public float SepalWidth;

            [Column("2")]
            public float PetalLength;

            [Column("3")]
            public float PetalWidth;

            [Column("4")]
            [ColumnName("Label")]
            public string Label = string.Empty;
        }

        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabel = string.Empty;
        }

        static void Main(string[] args)
        {
            var pipeline = new LearningPipeline();

            var dataPath = "data/iris-data.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            pipeline.Add(new Dictionarizer(nameof(IrisData.Label)));

            pipeline.Add(new ColumnConcatenator("Features", nameof(IrisData.SepalLength), nameof(IrisData.SepalWidth), nameof(IrisData.PetalLength), nameof(IrisData.PetalWidth)));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            var model = pipeline.Train<IrisData, IrisPrediction>();

            var dataLine = string.Empty;

            Console.WriteLine("Enter Data (Comma Separated): ");
            while(!String.IsNullOrWhiteSpace(dataLine = Console.ReadLine()))
            {
                var data = dataLine.Trim().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);

                if(data.Length != 4 || data.Any(d => !Regex.IsMatch(d.Trim(), @"\d+(\.\d+)?")))
                {
                    Console.WriteLine("Data must have 4 comma separated float values, please enter data again: ");
                }

                var prediction = model.Predict(new IrisData
                {
                    SepalLength = float.Parse(data[0].Trim()),
                    SepalWidth = float.Parse(data[1].Trim()),
                    PetalLength = float.Parse(data[2].Trim()),
                    PetalWidth = float.Parse(data[3].Trim())
                });

                Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabel}");
                Console.WriteLine("Re-enter Data (Comma Separated): ");
            }
        }
    }
}
