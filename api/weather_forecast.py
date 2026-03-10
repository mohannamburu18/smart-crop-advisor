"""
7-Day Weather Forecast with Farmer-Specific Alerts
Uses OpenWeatherMap forecast API
"""

import httpx
from datetime import datetime
from collections import defaultdict

ICONS = {'thunderstorm':'⛈️','drizzle':'🌦️','rain':'🌧️','snow':'❄️',
         'mist':'🌫️','fog':'🌫️','haze':'🌫️','dust':'💨','clear':'☀️','clouds':'⛅'}

def _emoji(desc):
    d = desc.lower()
    for k, v in ICONS.items():
        if k in d: return v
    return '🌤️'

def _alerts(d):
    alerts, level = [], 'ok'
    rain, tmax, tmin, hum, wind = d.get('rain_mm',0), d.get('temp_max',25), d.get('temp_min',15), d.get('humidity',60), d.get('wind_speed',0)
    if rain > 30:   alerts.append('Heavy rain — avoid spraying, check drainage'); level='danger'
    elif rain > 10: alerts.append('Moderate rain — good for soil moisture'); level='warn'
    if tmax > 38:   alerts.append('Extreme heat — water crops early morning'); level='danger'
    elif tmax > 33: alerts.append('Hot day — increase irrigation'); level = level if level=='danger' else 'warn'
    if tmin < 8:    alerts.append('Cold night — protect seedlings from frost'); level='danger'
    if hum > 85:    alerts.append('High humidity — watch for fungal diseases'); level = level if level in ['danger','warn'] else 'warn'
    if wind > 40:   alerts.append('Strong winds — support tall crops, delay spraying'); level = level if level in ['danger','warn'] else 'warn'
    if not alerts:  alerts.append('Good conditions for farming activities')
    return {'alerts': alerts, 'level': level}

async def get_7day_forecast(lat: float, lon: float, api_key: str) -> dict:
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&cnt=40"
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=10)
    if r.status_code != 200:
        raise Exception(f"Weather API error {r.status_code}")
    return _parse(r.json())

def _parse(data: dict) -> dict:
    daily = defaultdict(list)
    for item in data['list']:
        day = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
        daily[day].append(item)

    days = []
    for date_str, slots in list(daily.items())[:7]:
        dt    = datetime.strptime(date_str, '%Y-%m-%d')
        temps = [s['main']['temp'] for s in slots]
        hums  = [s['main']['humidity'] for s in slots]
        rains = [s.get('rain', {}).get('3h', 0) for s in slots]
        winds = [s['wind']['speed'] for s in slots]
        desc  = slots[len(slots)//2]['weather'][0]['description']
        d = {
            'date':        dt.strftime('%a, %d %b'),
            'emoji':       _emoji(desc),
            'description': desc.title(),
            'temp_max':    round(max(temps), 1),
            'temp_min':    round(min(temps), 1),
            'humidity':    round(sum(hums)/len(hums)),
            'rain_mm':     round(sum(rains), 1),
            'wind_speed':  round(max(winds) * 3.6, 1),
        }
        d.update(_alerts(d))
        days.append(d)

    return {
        'location':              data['city']['name'],
        'days':                  days,
        'best_day_to_spray':     _best_spray(days),
        'best_day_to_irrigate':  _best_irrigate(days),
    }

def _best_spray(days):
    best, bs = None, -999
    for d in days:
        s = 100 - d['rain_mm']*5 - d['humidity']*0.3 - d['wind_speed']*0.5
        if s > bs: bs, best = s, d['date']
    return best or 'Check daily'

def _best_irrigate(days):
    best, bs = None, -999
    for d in days:
        s = d['temp_max'] - d['rain_mm']*2
        if s > bs: bs, best = s, d['date']
    return best or 'Check daily'
