import pandas as pd
from sentiment_analysis import compute_sentiment, get_sentiment_series_all
from forecasting import prepare_weekly_forecast_all
from visualization import plot_all_products_subplots
from summary import generate_summary

df = pd.read_pickle("processed_reviews.pkl")

df = compute_sentiment(df)  
sentiment_dict = get_sentiment_series_all(df) 

forecast_results = prepare_weekly_forecast_all(df, periods=12, smooth_window=3)

product_classification = {}
for asin, data in forecast_results.items():
    forecast_series = data['forecast']['yhat']
    product_classification[asin] = "Positive" if forecast_series.mean() > 0 else "Negative"

plot_all_products_subplots(forecast_results, sentiment_dict, product_classification)

summary_df = generate_summary(
    products=forecast_results.keys(),
    forecast_results={k: v['forecast']['yhat'] for k, v in forecast_results.items()},
    classification_dict=product_classification,
    sentiment_dict=sentiment_dict
)

print("\nProduct Future Sentiment Summary:\n", summary_df.head())
