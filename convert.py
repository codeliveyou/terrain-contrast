

def convert(LA):
    latitude_deg = float(LA.split('°')[0])
    latitude_min = float(LA.split('°')[1].split("'")[0])
    latitude_sec = float(LA.split("'")[1])
    # Convert to decimal degrees
    decimal_latitude = latitude_deg + (latitude_min / 60) + (latitude_sec / 3600)
    
    return decimal_latitude