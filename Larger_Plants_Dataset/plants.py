import requests
import pandas as pd

species_list = [
    "Brasenia schreberi", "Cabomba", "Ceratophyllum demersum",
    "Elodea canadensis", "Heteranthera dubia", "Hydrocharis morsus-ranae",
    "Myriophyllum sibiricum", "Myriophyllum spicatum", "Najas flexilis",
    "Nitellopsis obtusa", "Nuphar variegata", "Nymphaea odorata",
    "Potamogeton crispus", "Potamogeton gramineus", "Potamogeton illinoensis",
    "Potamogeton natans", "Potamogeton praelongus", "Potamogeton richardsonii",
    "Potamogeton robbinsii", "Ranunculus aquatilis", "Vallisneria americana"
]

records = []

print("Starting direct GBIF API fetch for images...\n")

for name in species_list:
    try:
        # 1. Direct API call to match the taxonomy key
        match_url = f"https://api.gbif.org/v1/species/match?name={name}"
        match_data = requests.get(match_url).json()
        
        if match_data.get('matchType') != 'NONE' and 'usageKey' in match_data:
            key = match_data['usageKey']
            
            # 2. Fetch occurrences specifically filtering for StillImages
            search_url = f"https://api.gbif.org/v1/occurrence/search?taxonKey={key}&mediaType=StillImage&limit=100"
            search_data = requests.get(search_url).json()
            results = search_data.get('results', [])
            
            found_images = 0
            for r in results:
                # 3. Extract the actual image URL from the media array
                image_url = None
                for media_item in r.get('media', []):
                    if media_item.get('type') == 'StillImage':
                        image_url = media_item.get('identifier')
                        break
                        
                if image_url:
                    records.append({
                        'species': name,
                        'lat': r.get('decimalLatitude'),
                        'lon': r.get('decimalLongitude'),
                        'year': r.get('year'),
                        'image_url': image_url
                    })
                    found_images += 1
    except:
        pass
