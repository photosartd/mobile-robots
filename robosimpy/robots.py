import numpy as np

class Wheel:
    def __init__(self, alpha: float, l: float, beta, d: float, steerable: bool, motor: bool):
        """
        Stellt ein Rad des mobilen Roboters dar. Für die Konfiguration von alpha, beta, l und d siehe Skript.

        Parameters
        ----------
        alpha : float
            Winkel zum Gelenk
        l : float
            Länge zum Gelenk
        beta : float
            Winkel vom Gelenk zum Rad
        d : float
            Länge vom Gelenk zum Rad
        steerable : bool
            Ist das Rad drehbar/beweglich?
        motor : bool
            Ist das Rad angetrieben?
        """
        self.motor = motor
        self.steerable = steerable
        self.d = d
        self.beta = beta
        self.l = l
        self.alpha = alpha

class Robot:
    def __init__(self, state, initial_C_p, wheels, enclosure, lasers=None, wheel_radius=0.2):
        """
        Stellt den zu simulierenden Roboter dar.
        Enthält die Räder, Hülle und optional die Laser-Konfiguration und den Rad-Radius.

        Parameters
        ----------
        wheels : dict of str, Wheel
            Dictionary mit dem Radnamen (z.b. links, rechts, ... ) als Key und dem Wheel-Objekt als Value.
        enclosure : ndarray
            Hülle des mobilen Roboters im Format [[x1,y1,x2,y2],...].
            Wird zum Zeichnen des Roboters und für die Kollision benötigt.
        lasers : ndarray or None
            Winkel der Laser-Sensoren relativ zum Roboter.
        wheel_radius
            Bestimmt, wie große die Räder gezeichnet werden sollen.
        """

        self.state = state.astype(np.float64) # [x,y,theta] analog zur Vorlesung
        self.enclosure = np.array(enclosure)
        self.wheels = wheels
        self.lasers = np.empty(0) if lasers is None else np.array(lasers)
        self.wheel_radius = wheel_radius
        self.C_p = initial_C_p.astype(np.float64)

    @property
    def theta(self):
        return self.state[2]

    @theta.setter
    def theta(self, new_theta):
        self.state[2] = float(new_theta)

    @property
    def pos(self):
        return self.state[:2]

    @pos.setter
    def pos(self, new_pos):
        self.state[:2] = np.array(new_pos)