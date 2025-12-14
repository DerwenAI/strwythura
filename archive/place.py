

from icecream import ic
import placekey as pk

lat, lon = 0.0, 0.0

key = pk.geo_to_placekey(lat, lon)
ic(type(key))
ic(key)

ic(pk.placekey_to_geo(key))

h3 = pk.placekey_to_h3(key)
ic(type(h3))
ic(h3)

ic(pk.placekey_distance('@dvt-smp-tvz', '@5vg-7gq-tjv'))

print(pk.list_free_datasets())
print(pk.return_free_datasets_location_by_name('chipotle-locations'))
