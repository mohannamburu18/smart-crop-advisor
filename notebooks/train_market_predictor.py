"""Train Market Price Prediction Models using Prophet"""
import pandas as pd, numpy as np, joblib, json, os, warnings
warnings.filterwarnings('ignore')
from prophet import Prophet

DATA  = os.path.join(os.path.dirname(__file__), "../data/market_prices_historical.csv")
MDIR  = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MDIR, exist_ok=True)

df = pd.read_csv(DATA); df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df)} rows | Crops: {df['crop'].unique()}")

models, forecasts = {}, {}
for crop in df['crop'].unique():
    cdf = df[df['crop']==crop][['date','price_per_quintal']].copy()
    cdf.columns = ['ds','y']; cdf = cdf.sort_values('ds')
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode='multiplicative',
                changepoint_prior_scale=0.1, interval_width=0.8)
    m.fit(cdf)
    fut = m.make_future_dataframe(periods=6, freq='MS')
    fc  = m.predict(fut).tail(6)
    forecasts[crop] = [{'month': r['ds'].strftime('%b %Y'),
                        'price': round(float(r['yhat']),2),
                        'price_low': round(float(r['yhat_lower']),2),
                        'price_high': round(float(r['yhat_upper']),2)}
                       for _, r in fc.iterrows()]
    models[crop] = m
    best = fc.loc[fc['yhat'].idxmax()]
    print(f"  {crop:15s} → Best: {best['ds'].strftime('%b %Y')} @ ₹{best['yhat']:.0f}/q")

joblib.dump(models, f'{MDIR}/market_prophet_models.pkl')
with open(f'{MDIR}/market_forecasts_cache.json','w') as f:
    json.dump(forecasts, f, indent=2)
print("\n✅ market_prophet_models.pkl + market_forecasts_cache.json saved!")
