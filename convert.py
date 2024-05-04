
LA = "39°51'12.27"
LO = "105°14'6.60"

latitude_deg = float(LA.split('°')[0])
latitude_min = float(LA.split('°')[1].split("'")[0])
latitude_sec = float(LA.split("'")[1])

longitude_deg = float(LO.split('°')[0])
longitude_min = float(LO.split('°')[1].split("'")[0])
longitude_sec = float(LO.split("'")[1])

# Convert to decimal degrees
decimal_latitude = latitude_deg + (latitude_min / 60) + (latitude_sec / 3600)
decimal_longitude = longitude_deg + (longitude_min / 60) + (longitude_sec / 3600)

# Since the longitude is west, it should be negative
decimal_longitude = -decimal_longitude

print(decimal_latitude, decimal_longitude)


print((39.32 - 39.18) * (125.594 - 125.724) / (105.300 - 105.235) / (39.853 - 39.825))