namespace SentimentAnalysis
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.Models;
    using Microsoft.ML.Runtime.Api;
    using Microsoft.ML.Trainers;
    using Microsoft.ML.Transforms;
    using SentimentAnalysis.Models;

    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            var modelTask = TrainAsync();

            modelTask.Wait();

            var model = modelTask.Result;

            Evaluate(model);

            Predict(model);

            Console.ReadKey();
        }

        private static void Predict(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            IEnumerable<SentimentData> sentiments = new[] {
                new SentimentData{ SentimentText = "I'm very proud of what you have done."},
                new SentimentData{ SentimentText = "This is very nasty, I can't relay on this guy." },
                new SentimentData{ SentimentText = "Please refrain from adding nonsense to Wikipedia." },
                new SentimentData{ SentimentText = "He is the best, and the article should say that." },
                new SentimentData{ SentimentText = "Nice article." },
                new SentimentData{ SentimentText = "Not that good." },
                new SentimentData{ SentimentText = "Somehow good." },
                new SentimentData{ SentimentText = "I will not recommand this to anybody." },
            };

            var predictions = model.Predict(sentiments);

            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach(var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
        }

        static async Task<PredictionModel<SentimentData, SentimentPrediction>> TrainAsync()
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());

            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 13, NumTrees = 13, MinDocumentsInLeafs = 8 });

            var model = pipeline.Train<SentimentData, SentimentPrediction>();

            await model.WriteAsync(_modelPath);

            return model;
        }

        static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();
            var evaluator = new BinaryClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation.");
            Console.WriteLine("-------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }
    }
}
