import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # für 3D-Darstellung
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from matplotlib.widgets import Slider

# =============================================================================
# Globale Parameter für interaktive Steuerung
# =============================================================================
wind_x = 2.0       # Windgeschwindigkeit in X (m/s)
wind_y = 0.0       # Windgeschwindigkeit in Y (m/s)
turbulence_level = 0.2  # Standardabweichung für Turbulenzeffekte

# =============================================================================
# Globale Liste für Telemetrie-Daten (für Excel-Aufzeichnung)
# =============================================================================
telemetry_data = []
recording_finished = False

# Globale Listen für Zusatzplots (Höhe vs. Zeit, Treibstoff vs. Zeit)
time_data = []
altitude_data = []
fuel_data = []

# =============================================================================
# Rotationsmatrix aus Euler-Winkeln (Roll, Pitch, Yaw in Grad)
# =============================================================================
def get_rotation_matrix(attitude):
    # attitude: [roll, pitch, yaw] in Grad
    phi = np.deg2rad(attitude[0])
    theta = np.deg2rad(attitude[1])
    psi = np.deg2rad(attitude[2])
    
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1]])
    # Reihenfolge: R = R_z @ R_y @ R_x
    R = R_z @ R_y @ R_x
    return R

# =============================================================================
# Booster-Klasse: Erweiterte Dynamik, Sensorik, Guidance und Steuerung
# =============================================================================
class Booster:
    def __init__(self):
        # Physikalische Konstanten und Parameter
        self.mass = 20000.0             # Booster-Masse (kg)
        self.g = 9.81                   # Erdbeschleunigung (m/s²)
        self.fuel = 5000.0              # Anfangs-Treibstoff (kg)
        self.max_thrust = 1.5e6         # Maximale Schubkraft (N)
        self.Isp = 300.0                # Spezifischer Impuls (s)
        
        # Startzustand: Position und Geschwindigkeit im 3D-Raum
        # Das Landing Pad liegt immer bei (0,0,0)
        self.pos = np.array([300.0, -150.0, 1200.0], dtype=float)   # in Metern
        self.vel = np.array([-50.0, 30.0, -80.0], dtype=float)        # in m/s
        
        # Guidance-Parameter
        self.Vh_max = 50.0      # Maximale horizontale Geschwindigkeit (m/s)
        self.d_const = 100.0    # Charakteristische Länge für den horizontalen Übergang
        
        self.Vz_max = 50.0      # Maximale vertikale (Abstiegs-)Geschwindigkeit (m/s)
        self.z_const = 200.0    # Charakteristische Höhe für den vertikalen Übergang
        self.h_touch = 5.0      # Unterhalb dieser Höhe wird der vertikale Sollwert auf 0 gesetzt
        
        # PD-Regler-Koeffizienten
        self.kp_h = 0.1       # Horizontal: Reaktion auf den Geschwindigkeitsfehler
        self.kd_h = 0.05      # Horizontal: Dämpfung basierend auf dem Positionsfehler (Ziel = (0,0))
        self.kp_v = 0.8       # Vertikal: Reaktion auf den Geschwindigkeitsfehler
        self.kd_v = 0.3       # Vertikal: Dämpfung – in Terminal Descent zusätzlich Positionsfehler
        
        # Erweiterte Sensorik: Orientation (Roll, Pitch, Yaw) und Winkelgeschwindigkeit
        self.attitude = np.array([0.0, 0.0, 0.0], dtype=float)  
        self.angular_velocity = np.zeros(3, dtype=float)
        
        # Dynamisch berechneter Soll-Zustand (target velocity)
        self.target_vel = np.zeros(3, dtype=float)
        
        # Statusvariablen
        self.phase = "Approach"  
        self.time = 0.0          # Simulierte Zeit (s)

    def update_target(self):
        """
        Berechnet die Zielgeschwindigkeiten (target_vel) für eine glatte Guidance,
        sodass der Booster direkt auf das Landing Pad (0,0,0) zusteuert.
        Zusätzlich prüft sie, ob die Bedingungen zu extrem sind – bei zu starkem Wind
        oder Turbulenz wird die Phase auf "Abort Landing" gesetzt.
        """
        global wind_x, wind_y, turbulence_level
        if abs(wind_x) > 8 or abs(wind_y) > 8 or turbulence_level > 1.0:
            self.phase = "Abort Landing"
            self.target_vel = np.array([0.0, 0.0, -0.5])
            return
        
        pos_xy = self.pos[:2]
        d_h = np.linalg.norm(pos_xy)
        if d_h > 1e-6:
            Vh_target = self.Vh_max * np.tanh(d_h / self.d_const)
            target_xy = -Vh_target * (pos_xy / d_h)
        else:
            target_xy = np.array([0.0, 0.0], dtype=float)
        
        z = self.pos[2]
        if z > self.h_touch:
            Vz_target = self.Vz_max * np.tanh(z / self.z_const)
            target_z = -Vz_target
        else:
            target_z = 0.0
        
        self.target_vel = np.array([target_xy[0], target_xy[1], target_z], dtype=float)
        
        if d_h > 10.0 or self.pos[2] > 10.0:
            self.phase = "Approach"
        elif self.pos[2] > 3.0 or d_h > 3.0:
            self.phase = "Terminal Descent"
        else:
            self.phase = "Touchdown"

    def compute_control(self):
        """
        Berechnet mittels PD-Regelung den erforderlichen Beschleunigungsvektor (a_command).
        """
        self.update_target()
        error_vel_h = self.target_vel[:2] - self.vel[:2]
        error_pos_h = -self.pos[:2]
        a_command_h = self.kp_h * error_vel_h + self.kd_h * error_pos_h
        
        error_vel_v = self.target_vel[2] - self.vel[2]
        if self.pos[2] < 20.0:
            error_pos_v = (0.0 - self.pos[2])
            a_command_v = self.kp_v * error_vel_v + self.kd_v * error_pos_v
        else:
            a_command_v = self.kp_v * error_vel_v
        
        a_command = np.array([a_command_h[0], a_command_h[1], a_command_v], dtype=float)
        return a_command

    def compute_thrust(self, a_command):
        """
        Berechnet die notwendige Triebwerkskraft T, sodass gilt:
            a = T/m + g  -->  T = m*(a_command - g_vector)
        mit g_vector = [0, 0, -g].
        """
        g_vector = np.array([0.0, 0.0, -self.g], dtype=float)
        T = self.mass * (a_command - g_vector)
        thrust_mag = np.linalg.norm(T)
        if thrust_mag > self.max_thrust:
            T = T * (self.max_thrust / thrust_mag)
        return T

    def update(self, dt):
        """
        Führt einen Zeitschritt dt aus und aktualisiert:
         • Translation (Position, Geschwindigkeit)
         • Erweiterte Sensorik (Attitude, Winkel)
         • Wind- und Turbulenzeffekte
         • Momentane G-Kraft
         • Bei Bodenkontakt erfolgt ein Touchdown
         
         Rückgabe: pos, vel, thrust, attitude, g_force, turbulence_magnitude
        """
        self.time += dt
        
        if self.fuel > 0:
            a_command = self.compute_control()
            thrust = self.compute_thrust(a_command)
        else:
            thrust = np.array([0.0, 0.0, 0.0], dtype=float)
        
        g_vector = np.array([0.0, 0.0, -self.g], dtype=float)
        
        # Wind- und Turbulenzeffekte
        global wind_x, wind_y, turbulence_level
        wind_effect = np.array([wind_x, wind_y, 0.0], dtype=float)
        turbulence = np.array([np.random.normal(scale=turbulence_level),
                               np.random.normal(scale=turbulence_level),
                               np.random.normal(scale=turbulence_level/2)])
        turbulence_magnitude = np.linalg.norm(turbulence)
        
        acceleration = thrust / self.mass + g_vector + wind_effect + turbulence
        
        self.vel += acceleration * dt
        self.pos += self.vel * dt
        
        consumption_rate = np.linalg.norm(thrust) / (self.Isp * self.g)
        self.fuel -= consumption_rate * dt
        if self.fuel < 0.0:
            self.fuel = 0.0
        
        if self.pos[2] <= 0:
            self.pos[2] = 0.0
            self.vel = np.zeros(3, dtype=float)
            thrust = np.zeros(3, dtype=float)
            self.fuel = 0.0
            self.phase = "Touchdown"
        
        # Simuliere einfache Winkelkorrektur (Attitude)
        self.angular_velocity = -0.1 * self.attitude + np.random.normal(scale=0.05, size=3)
        self.attitude += self.angular_velocity * dt
        
        g_force = np.linalg.norm(acceleration) / self.g
        
        return (self.pos.copy(), self.vel.copy(), thrust.copy(),
                self.attitude.copy(), g_force, turbulence_magnitude)

# =============================================================================
# Neue Funktion: Zeichnen des rotierenden Boosters (als Zylinder)
# =============================================================================
def draw_booster(ax, pos, attitude):
    booster_height = 50.0
    booster_radius = 1.85
    resolution = 20
    
    # Bestimme die Rotationsmatrix aus den Attitude-Werten (in Grad)
    R = get_rotation_matrix(attitude)
    
    # Erzeuge lokale Punkte (im Booster-Koordinatensystem)
    theta = np.linspace(0, 2 * np.pi, resolution)
    bottom = np.vstack((booster_radius * np.cos(theta),
                        booster_radius * np.sin(theta),
                        np.zeros_like(theta))).T  # shape (resolution, 3)
    top = bottom.copy()
    top[:, 2] += booster_height
    
    verts = []
    for i in range(resolution - 1):
        pts = np.vstack((bottom[i], bottom[i+1], top[i+1], top[i]))
        # Transformation: Rotation und dann Translation
        pts_rot = pts.dot(R.T) + pos  # R.T weil wir Zeilen als Vektoren interpretieren
        verts.append(pts_rot.tolist())
    pts = np.vstack((bottom[-1], bottom[0], top[0], top[-1]))
    pts_rot = pts.dot(R.T) + pos
    verts.append(pts_rot.tolist())
    
    booster_patch = Poly3DCollection(verts, facecolors='silver', edgecolors='gray', alpha=0.8)
    ax.add_collection3d(booster_patch)
    return booster_patch

# =============================================================================
# Grafikfunktionen: Zeichnen des Landing Pads (roter Kreis)
# =============================================================================
def draw_landing_pad(ax):
    pad_radius = 10.0
    theta = np.linspace(0, 2 * np.pi, 50)
    x = pad_radius * np.cos(theta)
    y = pad_radius * np.sin(theta)
    z = np.zeros_like(theta)
    ax.plot(x, y, z, 'r-', lw=3)
    ax.plot_trisurf(x, y, z, color='red', alpha=0.2)

# =============================================================================
# Interaktive Steuerung: Slider für Wind und Turbulenz
# =============================================================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.30)

axwind_x = fig.add_axes([0.25, 0.20, 0.50, 0.03])
axwind_y = fig.add_axes([0.25, 0.15, 0.50, 0.03])
axturb = fig.add_axes([0.25, 0.10, 0.50, 0.03])

slider_wind_x = Slider(axwind_x, 'Wind X', -10.0, 10.0, valinit=2.0)
slider_wind_y = Slider(axwind_y, 'Wind Y', -10.0, 10.0, valinit=0.0)
slider_turb = Slider(axturb, 'Turbulence', 0.0, 2.0, valinit=0.2)

def update_wind_x(val):
    global wind_x
    wind_x = slider_wind_x.val
def update_wind_y(val):
    global wind_y
    wind_y = slider_wind_y.val
def update_turb(val):
    global turbulence_level
    turbulence_level = slider_turb.val

slider_wind_x.on_changed(update_wind_x)
slider_wind_y.on_changed(update_wind_y)
slider_turb.on_changed(update_turb)

# =============================================================================
# Zusatzplots: Zweites Fenster für "Höhe vs. Zeit" und "Treibstoff vs. Zeit"
# =============================================================================
fig2, (ax_alt, ax_fuel) = plt.subplots(2, 1, figsize=(8, 6))
ax_alt.set_title("Höhe (z) vs. Zeit")
ax_alt.set_xlabel("Zeit (s)")
ax_alt.set_ylabel("Höhe (m)")
ax_fuel.set_title("Treibstoff vs. Zeit")
ax_fuel.set_xlabel("Zeit (s)")
ax_fuel.set_ylabel("Treibstoff (kg)")
alt_line, = ax_alt.plot([], [], 'b-')
fuel_line, = ax_fuel.plot([], [], 'g-')

# =============================================================================
# Kameraeinstellungen und 3D-Plot (Ansicht vom Landing Pad)
# =============================================================================
ax.set_xlim(-400, 400)
ax.set_ylim(-400, 400)
ax.set_zlim(0, 1400)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Höhe (m)")
ax.set_title("SpaceX Booster Landing – Ansicht vom Landing Pad")
ax.view_init(elev=20, azim=160)
ax.dist = 7
draw_landing_pad(ax)

flight_path_line, = ax.plot([], [], [], 'b-', lw=2, label='Flugbahn')
# Info-Text unten (0.02, 0.85)
info_text = fig.text(0.02, 0.85, "", fontsize=10)

# =============================================================================
# Globale Variablen für Animation und Aufzeichnung
# =============================================================================
dt = 0.05  # 50 ms pro Frame (Echtzeit)
booster = Booster()
trajectory = []
booster_patch = None
recording_finished = False

def init_animation():
    global booster_patch, trajectory, time_data, altitude_data, fuel_data
    trajectory.clear()
    flight_path_line.set_data([], [])
    flight_path_line.set_3d_properties([])
    info_text.set_text("")
    time_data.clear()
    altitude_data.clear()
    fuel_data.clear()
    if booster_patch is not None:
        booster_patch.remove()
    booster_patch = draw_booster(ax, booster.pos, booster.attitude)
    return [booster_patch, flight_path_line, info_text]

def animate(frame):
    global booster_patch, trajectory, booster, telemetry_data, recording_finished, time_data, altitude_data, fuel_data
    pos, vel, thrust, attitude, g_force, turb_mag = booster.update(dt)
    trajectory.append(pos.copy())
    
    traj_arr = np.array(trajectory)
    flight_path_line.set_data(traj_arr[:, 0], traj_arr[:, 1])
    flight_path_line.set_3d_properties(traj_arr[:, 2])
    if booster_patch is not None:
        booster_patch.remove()
    booster_patch = draw_booster(ax, booster.pos, booster.attitude)
    
    # Erzeuge bedingte Meldungen (inklusive Triebwerksstatus)
    msgs = []
    if booster.phase == "Touchdown":
        msgs.append("Touchdown erreicht!")
    if booster.phase == "Abort Landing":
        msgs.append("Landing aborted – unsafe conditions!")
    if turb_mag > 1.5:
        msgs.append(f"Hohe Turbulenzen: {turb_mag:.2f} m/s")
    if abs(wind_x) > 8 or abs(wind_y) > 8:
        msgs.append("Starker Wind erkannt!")
    horiz_dist = np.linalg.norm(booster.pos[:2])
    if horiz_dist < 20:
        msgs.append(f"Geringe horizontale Abweichung: {horiz_dist:.2f} m")
    if booster.phase == "Terminal Descent":
        msgs.append("Terminal Descent")
    if g_force > 3.0:
        msgs.append(f"Hohe G-Kraft: {g_force:.2f} g")
    thrust_norm = np.linalg.norm(thrust)
    engine_percent = (thrust_norm / booster.max_thrust) * 100
    if thrust_norm > 1e-3:
        msgs.append(f"Gegenschub aktiv (Triebwerke: {engine_percent:.1f}%)")
    extra_msg = " | ".join(msgs)
    
    info_text.set_text(
        f"Zeit: {booster.time:.2f} s\n"
        f"Phase: {booster.phase}\n"
        f"Pos: ({booster.pos[0]:.2f}, {booster.pos[1]:.2f}, {booster.pos[2]:.2f}) m\n"
        f"Vel: ({booster.vel[0]:.2f}, {booster.vel[1]:.2f}, {booster.vel[2]:.2f}) m/s\n"
        f"Treibstoff: {booster.fuel:.2f} kg\n"
        f"TargetVel: ({booster.target_vel[0]:.2f}, {booster.target_vel[1]:.2f}, {booster.target_vel[2]:.2f}) m/s\n"
        f"Attitude (Roll,Pitch,Yaw): ({attitude[0]:.2f}°, {attitude[1]:.2f}°, {attitude[2]:.2f}°)\n"
        f"G-Force: {g_force:.2f} g\n"
        f"Meldungen: {extra_msg}"
    )
    
    telemetry_data.append({
        "Zeit (s)": booster.time,
        "Phase": booster.phase,
        "PosX (m)": booster.pos[0],
        "PosY (m)": booster.pos[1],
        "PosZ (m)": booster.pos[2],
        "VelX (m/s)": booster.vel[0],
        "VelY (m/s)": booster.vel[1],
        "VelZ (m/s)": booster.vel[2],
        "Treibstoff (kg)": booster.fuel,
        "TargetVelX (m/s)": booster.target_vel[0],
        "TargetVelY (m/s)": booster.target_vel[1],
        "TargetVelZ (m/s)": booster.target_vel[2],
        "Roll (deg)": attitude[0],
        "Pitch (deg)": attitude[1],
        "Yaw (deg)": attitude[2],
        "G-Force": g_force,
        "Turbulence (m/s)": turb_mag,
        "Engine (%)": engine_percent,
        "Message": extra_msg
    })
    
    time_data.append(booster.time)
    altitude_data.append(booster.pos[2])
    fuel_data.append(booster.fuel)
    alt_line.set_data(time_data, altitude_data)
    ax_alt.relim()
    ax_alt.autoscale_view()
    fuel_line.set_data(time_data, fuel_data)
    ax_fuel.relim()
    ax_fuel.autoscale_view()
    fig2.canvas.draw_idle()
    
    if (booster.phase in ["Touchdown", "Abort Landing"]) and not recording_finished:
        recording_finished = True
        telemetry_data.append({
            "Zeit (s)": booster.time,
            "Phase": "Aufzeichnung beendet – " + booster.phase,
            "PosX (m)": booster.pos[0],
            "PosY (m)": booster.pos[1],
            "PosZ (m)": booster.pos[2],
            "VelX (m/s)": booster.vel[0],
            "VelY (m/s)": booster.vel[1],
            "VelZ (m/s)": booster.vel[2],
            "Treibstoff (kg)": booster.fuel,
            "TargetVelX (m/s)": booster.target_vel[0],
            "TargetVelY (m/s)": booster.target_vel[1],
            "TargetVelZ (m/s)": booster.target_vel[2],
            "Roll (deg)": attitude[0],
            "Pitch (deg)": attitude[1],
            "Yaw (deg)": attitude[2],
            "G-Force": g_force,
            "Turbulence (m/s)": turb_mag,
            "Engine (%)": engine_percent,
            "Message": "Aufzeichnung beendet – " + booster.phase
        })
        df = pd.DataFrame(telemetry_data)
        df.to_excel("telemetry_data.xlsx", index=False)
        print("Telemetrie-Daten in 'telemetry_data.xlsx' gespeichert.")
        ani.event_source.stop()
    
    return [booster_patch, flight_path_line, info_text]

ani = FuncAnimation(fig, animate, frames=1000, init_func=init_animation, interval=50, blit=False)
plt.show()
plt.show(block=False)
