﻿namespace TaxiFarePrediction
{
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.Models;
    using Microsoft.ML.Trainers;
    using Microsoft.ML.Transforms;
    using System;
    using System.IO;
    using System.Threading.Tasks;
    using TaxiFarePrediction.Models;

    class Program
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            var model = Train();

            Evaluate(model);

            Predict(model);

            Console.Read();
        }

        private static void Predict(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var prediction = model.Predict(TestTrips.Trip1);

            Console.WriteLine();
            Console.WriteLine("Taxi Fare Prediction");
            Console.WriteLine("--------------------");
            Console.WriteLine("Predicted fare {0}, actual fare 29.5", prediction.FareAmount);
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();

            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"Rms = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");
        }

        private static PredictionModel<TaxiTrip, TaxiTripFarePrediction> Train()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','),
                new ColumnCopier(("FareAmount", "Label")),
                new CategoricalOneHotVectorizer(
                    "VendorId",
                    "RateCode",
                    "PaymentType"),
                new ColumnConcatenator(
                    "Features",
                    "VendorId",
                    "RateCode",
                    "PassengerCount",
                    "TripDistance",
                    "PaymentType"),
                new FastTreeRegressor()
            };

            var model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();

            model.WriteAsync(_modelpath).Wait();

            return model;
        }
    }
}
