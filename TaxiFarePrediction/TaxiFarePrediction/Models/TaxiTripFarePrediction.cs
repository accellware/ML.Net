using Microsoft.ML.Runtime.Api;

namespace TaxiFarePrediction.Models
{

    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
