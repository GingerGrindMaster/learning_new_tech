import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime
from keycloak import KeycloakOpenID
import httpx
"""
POUZIVANI APLIKACE:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
- pro získání tokenu a jeho pouziti
curl -X POST http://localhost:8000/token
curl -X GET http://localhost:8000/wantedendpoint -H "Authorization: Bearer <tvůj_token>"
"""
app = FastAPI()

WEATHER_APIKEY= "5524adf94247849adf5d44f79dd413db"
GEOCODING_APIKEY= "0f5ac8e532584b8f95381a0b69fb0cb5"

# Configure Keycloak OpenID client
keycloak_openid = KeycloakOpenID(server_url="http://localhost:8080/",  # keycloak server
                                 client_id="weather-api-client",  # client-id created in keycloak
                                 realm_name="myrealm",  # realm-name created in keycloak
                                 client_secret_key="pzEhyXaPd55IgzzxTuj3lGeJqSIMYhBP")  # client-secret created in keycloak

# Definice OAuth2 schématu pro získání tokenu
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def get_token():
    # Získání tokenu pomocí client credentials
    token = keycloak_openid.token(grant_type='client_credentials')
    return token

@app.get("/")
def root_endpoint():
    return {"message": "Welcome to weather API"}

#
@app.get("/time")
def show_time(token: str = Depends(oauth2_scheme)):
    """
    Když je endpoint volán, FastAPI automaticky zavolá funkci oauth2_scheme,
    která se pokusí extrahovat token z hlavičky Authorization v příchozím požadavku.
    Depends(oauth2_scheme), říkáte FastAPI, že tento endpoint potřebuje ověřený token.
    """
    try:
        # Dekóduj token a ověř jeho platnost
        token_info = keycloak_openid.decode_token(token, validate=True)
        print("Given token is valid")

        # Pokud je token platný, vrátí aktuální čas
        return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        # Pokud došlo k chybě (např. token neplatný), vyhoď HTTP výjimku
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/geocode")
async def show_geocode(city: str):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={city}&key={GEOCODING_APIKEY}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()

    if response.status_code == 200:
        return {
            "latitude": data["results"][0]["geometry"]["lat"],
            "longitude": data["results"][0]["geometry"]["lng"]
        }


@app.get("/city_weather")
async def show_city_weather(city: str):
    url_city_location = f"https://api.opencagedata.com/geocode/v1/json?q={city}&key={GEOCODING_APIKEY}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url_city_location)
        data = response.json()
    if response.status_code == 200:
        # Return the first result, which contains lat and lon
        lat = data["results"][0]["geometry"]["lat"]
        lon = data["results"][0]["geometry"]["lng"]
        #print("lat: ", lat, "lon: ", lon)

    url_weather = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_APIKEY}"
    async with httpx.AsyncClient() as client:  # async client, like requests but for async
        response = await client.get(url_weather)

    if response.status_code == 200:
        w_data = response.json()
        result = {
            "weather_main": w_data["weather"][0]["main"],
            "weather_description": w_data["weather"][0]["description"],
            "temperature": w_data["main"]["temp"],
            "pressure": w_data["main"]["pressure"],
            "humidity": w_data["main"]["humidity"],
            "wind_speed": w_data["wind"]["speed"],
            "wind_deg": w_data["wind"]["deg"],
            "country": w_data["sys"]["country"],
            "name": w_data["name"]
        }
        return result
    else:
        return {"error": "Could not retrieve weather data."}

@app.get("/weather")
async def show_weather(lat: float, lon: float):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_APIKEY}"

    async with httpx.AsyncClient() as client:  # async client, like requests but for async
        response = await client.get(url)

    if response.status_code == 200:
        data = response.json()
        result= {
            "weather_main": data["weather"][0]["main"],
            "weather_description": data["weather"][0]["description"],
            "temperature": data["main"]["temp"],
            "pressure": data["main"]["pressure"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "wind_deg": data["wind"]["deg"],
            "country": data["sys"]["country"],
            "name": data["name"]
        }
        with open("weather_data.txt", "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, ensure_ascii=False)
        return result
    else:
        return {"error": "Could not retrieve weather data."}