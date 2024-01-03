import RPi.GPIO as g

g.setmode(g.BOARD)
g.setwarnings(False)
g.setup(38, g.IN)

while True:
    print(g.input(38))