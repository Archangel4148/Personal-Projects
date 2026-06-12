import python_weather

import asyncio


async def main() -> None:
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
        # Fetch a weather forecast from a city.
        location = "Rolla, Missouri"
        weather = await client.get(location)
        print(f"Weather for {location}:")
        print(f"{'Condition:':<16}", weather.description)
        print(f"{'Temperature:':<16}", weather.temperature, "°F")
        print(f"{'Feels Like:':<16}", weather.feels_like, "°F")
        print(f"{'Humidity:':<16}", weather.humidity, "%")
        print(f"{'Wind Speed:':<16}", weather.wind_speed, "mph")
        print(f"{'Wind Direction:':<16}", weather.wind_direction)
        print(f"{'Precipitation:':<16}", weather.precipitation, "in")
        print(f"{'Pressure:':<16}", weather.pressure, "inHg")


if __name__ == '__main__':
    asyncio.run(main())