import time, serial, sys
from skyfield.api import N, S, E, W, wgs84, load

rotator = serial.Serial(
    port=sys.argv[1],
    baudrate=9600,
    bytesize=8,
    parity='N',
    stopbits=1,
    timeout=1.0,
)

def rot_move(alt, az):
    rotator.write(f'm {alt:.5f} {az:.5f}'.encode('ascii'))
    rotator.flush()
    while not len(rotator.read_until()): pass

def rot_set(alt, az):
    rotator.write(f's {alt:.5f} {az:.5f}'.encode('ascii'))
    rotator.flush()
    while not len(rotator.read_until()): pass

def rot_get():
    rotator.write('c'.encode('ascii'))
    alt = rotator.read_until().decode('ascii').strip()
    az = rotator.read_until().decode('ascii').strip()
    return f'{alt} {az}'

ts = load.timescale()
eph = load('de421.bsp')
sun, earth = eph['sun'], eph['earth']
loc = earth + wgs84.latlon(45.27618 * N, 13.72136 * W, elevation_m=254)

while True:
    time.sleep(1.0)
    pos = loc.at(ts.now()).observe(sun).apparent()

    alt, az, dist = pos.altaz()
    ha, dec, dist = pos.hadec()

    print(rot_get())
    rot_set(alt.degrees, az.degrees)
    rot_move(alt.degrees, az.degrees)
