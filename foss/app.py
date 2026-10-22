import logging
import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from forecasting import prepare_weekly_forecast_all
from sentiment_analysis import get_sentiment_series_all
from visualization import plot_product_forecast_subplots  
from summary import analyze_series 
from accuracy_check import run_classification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

os.makedirs("static", exist_ok=True)

forecast_results = {}  
sentiment_dict = {}     

asin_to_name = {
    "B00007GDFV": "phone case",
    "B00062NHH0": "shirt",
    "B00066G516": "socks",
    "B0006HB4XE": "chain armament",
    "B0007MV6PO": "hat",
    "B0007MV6Q8": "cap",
    "B0007MV6PY": "red hat",
    "B00080L00Q": "kit",
}

name_to_asin = {v.lower(): k for k, v in asin_to_name.items()}

try:
    df = pd.read_csv("amazon_reviews.csv")
    known_asins = set(df["asin"].dropna().astype(str).str.strip().str.upper())
    log.info(f"Loaded {len(known_asins)} unique ASINs from amazon_reviews.csv")
except Exception as e:
    log.warning(f"Could not load amazon_reviews.csv at startup: {e}. Falling back to static mapping.")
    known_asins = set(asin_to_name.keys())

def api_error(message, code=400):
    return jsonify({"error": str(message)}), code

def normalize_asin(s):
    return s.strip().upper() if isinstance(s, str) and s.strip() else None

def extract_asin():
    asin = None
    if request.method == "GET":
        asin = request.args.get("asin") or request.args.get("product_name")
    elif request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "asin" in data and data["asin"]:
            asin = data["asin"]
        elif "product_name" in data and data["product_name"]:
            pn = data["product_name"].strip().lower()
            asin = name_to_asin.get(pn)
    if asin and not asin.upper().startswith("B"):
        maybe = str(asin).strip().lower()
        asin_mapped = name_to_asin.get(maybe)
        if asin_mapped:
            return normalize_asin(asin_mapped)
    return normalize_asin(asin) if asin else None

def init_for_asin(asin):

    asin = normalize_asin(asin)
    if not asin:
        raise ValueError("ASIN is missing")

    try:
        reviews = pd.read_csv("amazon_reviews.csv")
    except Exception as e:
        log.exception("Failed to read amazon_reviews.csv")
        raise FileNotFoundError("amazon_reviews.csv not found or unreadable") from e

    reviews['asin'] = reviews['asin'].astype(str)
    mask = reviews['asin'].str.strip().str.upper() == asin
    subset = reviews[mask]
    if subset.empty:
        raise ValueError(f"No reviews found in CSV for ASIN '{asin}'")

    try:
        forecast_dict = prepare_weekly_forecast_all(subset)
    except Exception as e:
        log.exception("Forecast computation failed for ASIN %s", asin)
        raise RuntimeError(f"Forecast computation failed for ASIN {asin}: {e}")

    try:
        sentiment_series_dict = get_sentiment_series_all(subset)
    except Exception as e:
        log.exception("Sentiment computation failed for ASIN %s", asin)
        raise RuntimeError(f"Sentiment computation failed for ASIN {asin}: {e}")

    for k, val in forecast_dict.items():
        k_up = normalize_asin(k)
        if 'name' not in val or not val['name']:
            val['name'] = asin_to_name.get(k_up, k_up)
        forecast_results[k_up] = val
        asin_to_name[k_up] = val.get('name', asin_to_name.get(k_up, k_up))
        name_to_asin[val['name'].lower()] = k_up

    for k, series in sentiment_series_dict.items():
        k_up = normalize_asin(k)
        sentiment_dict[k_up] = series

    log.info("Initialized ASIN %s (forecast entries: %d, sentiment entries: %d)",
             asin, len(forecast_dict), len(sentiment_series_dict))
    return True

def ensure_initialized(asin):
    asin = normalize_asin(asin)
    if not asin:
        raise ValueError("ASIN is missing")

    if asin in forecast_results and asin in sentiment_dict:
        return True

    if asin not in known_asins and asin not in asin_to_name:
        try:
            init_for_asin(asin)
            known_asins.add(asin)
            return True
        except Exception as e:
            raise ValueError(f"Unknown ASIN '{asin}' and initialization failed: {e}")

    try:
        init_for_asin(asin)
        known_asins.add(asin)
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to initialize data for ASIN {asin}: {e}")

@app.route("/")
def home():
    return {
        "message": "Future Product Prediction API is running!",
        "available_endpoints": [
            "/products",
            "/init_all (POST)",
            "/init_asin (POST)",
            "/product_summary (POST)",
            "/product_visualization (POST)",
            "/forecast (GET/POST)",
            "/sentiment (GET/POST)"
        ]
    }, 200

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/products", methods=["GET"])
def list_products():
    products = []
    combined_asins = set(list(asin_to_name.keys()) + list(known_asins))
    for a in sorted(combined_asins):
        products.append({"asin": a, "name": asin_to_name.get(a, "")})
    return jsonify({"count": len(products), "products": products})

@app.route("/init_all", methods=["POST"])
def init_all():
    try:
        reviews = pd.read_csv("amazon_reviews.csv")
    except Exception as e:
        log.exception("init_all failed to read CSV")
        return api_error("Could not read amazon_reviews.csv", 500)

    try:
        fdict = prepare_weekly_forecast_all(reviews)
        sdict = get_sentiment_series_all(reviews)

        forecast_results.clear()
        sentiment_dict.clear()
        for k, v in fdict.items():
            k_up = normalize_asin(k)
            forecast_results[k_up] = v
            asin_to_name[k_up] = v.get("name", asin_to_name.get(k_up, k_up))
            name_to_asin[asin_to_name[k_up].lower()] = k_up
            known_asins.add(k_up)

        for k, series in sdict.items():
            k_up = normalize_asin(k)
            sentiment_dict[k_up] = series

        log.info("Initialized all products from CSV: %d forecasts, %d sentiment series",
                 len(forecast_results), len(sentiment_dict))
        return jsonify({"status": "ok", "forecast_count": len(forecast_results), "sentiment_count": len(sentiment_dict)})
    except Exception as e:
        log.exception("init_all failed")
        return api_error(f"Initialization failed: {e}", 500)

@app.route("/init_asin", methods=["POST"])
def init_asin_endpoint():
    data = request.get_json(silent=True) or {}
    asin = data.get("asin") or (name_to_asin.get(data.get("product_name", "").strip().lower()) if data.get("product_name") else None)
    if not asin:
        return api_error("Missing 'asin' or 'product_name' in request", 400)
    try:
        init_for_asin(asin)
        return jsonify({"status": "initialized", "asin": asin}), 200
    except FileNotFoundError:
        return api_error("amazon_reviews.csv not found on server", 500)
    except Exception as e:
        log.exception("init_asin failed")
        return api_error(str(e), 500)

@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    asin = extract_asin()
    if not asin:
        return api_error("Missing 'asin' parameter or 'product_name'", 400)
    try:
        ensure_initialized(asin)

        entry = forecast_results.get(asin)
        if not entry or 'forecast' not in entry:
            return api_error("Forecast data not available for this ASIN", 404)

        forecast_df = entry["forecast"]
        out = forecast_df.loc[:, ["ds", "yhat"]].to_dict(orient="records")
        return jsonify({"asin": asin, "name": entry.get("name"), "forecast": out}), 200
    except ValueError as ve:
        return api_error(str(ve), 404)
    except Exception as e:
        log.exception("Error in /forecast")
        return api_error(str(e), 500)

@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    asin = extract_asin()
    if not asin:
        return api_error("Missing 'asin' parameter or 'product_name'", 400)
    try:
        ensure_initialized(asin)
        series = sentiment_dict.get(asin)
        if series is None:
            return api_error("Sentiment series not available for this ASIN", 404)
        sentiment_data = [
            {"date": str(idx), "score": float(val) if pd.notna(val) else None}
            for idx, val in series.items()
        ]
        return jsonify({"asin": asin, "sentiment": sentiment_data})
    except ValueError as ve:
        return api_error(str(ve), 404)
    except Exception as e:
        log.exception("Error in /sentiment")
        return api_error(str(e), 500)
       
@app.route("/classification_accuracy", methods=["GET"])
def classification_accuracy():
    try:
        print("Running classification...")
        result = run_classification("clean_reviews.csv")
        print("Classification finished:", result)
        return jsonify({"status": "ok", "metrics": result}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return api_error(str(e), 500)



@app.route("/product_summary", methods=["GET", "POST"])
def product_summary():
    asin = extract_asin()
    if not asin:
        return api_error("Missing 'asin' parameter or 'product_name'", 400)

    try:
        ensure_initialized(asin)

        series = sentiment_dict.get(asin)
        forecast_entry = forecast_results.get(asin)
        if series is None or forecast_entry is None:
            return api_error("ASIN found but not fully initialized", 404)

        hist_summary = analyze_series(series, label="Historical")

        forecast_df = forecast_entry.get("forecast")
        forecast_summary = None
        if forecast_df is not None and "yhat" in forecast_df:
            forecast_summary = analyze_series(forecast_df["yhat"], label="Forecast")
        else:
            forecast_summary = {"message": "No forecast data available"}

        def make_json_safe(obj):
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            if isinstance(obj, (pd.Timestamp, )):
                return str(obj)
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception:
                    return str(obj)
            if isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_json_safe(v) for v in obj]
            return obj

        return jsonify({
            "asin": asin,
            "product_name": forecast_entry.get("name"),
            "historical_summary": make_json_safe(hist_summary),
            "forecast_summary": make_json_safe(forecast_summary)
        })

    except ValueError as ve:
        return api_error(str(ve), 404)
    except Exception as e:
        log.exception("Error in /product_summary")
        return api_error(str(e), 500)

@app.route("/product_visualization", methods=["GET", "POST"])
def product_visualization():
    asin = extract_asin()
    if not asin:
        return api_error("Missing 'asin' parameter or 'product_name'", 400)
    try:
        ensure_initialized(asin)
        entry = forecast_results.get(asin)
        sentiment_series = sentiment_dict.get(asin)
        if not entry or sentiment_series is None:
            return api_error("ASIN found but not fully initialized", 404)

        fig_path = f"static/{asin}_plot.png"
        plot_product_forecast_subplots(asin, forecast_results, sentiment_series, save_path=fig_path)

        return jsonify({
            "asin": asin,
            "product_name": entry.get("name"),
            "forecast": entry["forecast"][["ds", "yhat"]].rename(columns={"ds": "date"}).to_dict(orient="records"),
            "sentiment": [
                {"date": str(idx), "score": float(val) if pd.notna(val) else None}
                for idx, val in sentiment_series.items()
            ],
            "image_path": fig_path
        })
    except ValueError as ve:
        return api_error(str(ve), 404)
    except Exception as e:
        log.exception("Error in /product_visualization")
        return api_error(str(e), 500)

@app.route("/graph/<path:product_name>", methods=["GET"])
def graph_by_name(product_name):
    """Fetch forecast + sentiment by product name directly (for frontend)."""
    if not product_name:
        return api_error("Missing product name", 400)

    product_name = product_name.strip().lower()
    asin = name_to_asin.get(product_name)
    if not asin:
        return api_error(f"Product '{product_name}' not found", 404)

    try:
        ensure_initialized(asin)
        forecast_entry = forecast_results.get(asin)
        sentiment_series = sentiment_dict.get(asin)

        if not forecast_entry or sentiment_series is None:
            return api_error(f"No data available for product '{product_name}'", 404)

        forecast_df = forecast_entry["forecast"].copy()
        last_hist_date = pd.to_datetime(sentiment_series.index).max()
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
        future_only = forecast_df[forecast_df["ds"] > last_hist_date]

        return jsonify({
            "asin": asin,
            "product_name": forecast_entry.get("name"),
            "forecast": future_only[["ds", "yhat"]]
                        .rename(columns={"ds": "date"}).to_dict(orient="records"),
            "sentiment": [
                {"date": str(idx), "score": float(val) if pd.notna(val) else None}
                for idx, val in sentiment_series.items()
            ]
        }), 200

    except Exception as e:
        log.exception("Error in /graph/<product_name>")
        return api_error(str(e), 500)



if __name__ == "__main__":
    log.info("Starting server (no automatic init). Use /init_all or /init_asin to prepare data on demand.")
    app.run(debug=True, port=5000)
