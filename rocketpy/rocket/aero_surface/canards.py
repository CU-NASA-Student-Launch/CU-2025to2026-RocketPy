from .generic_surface import GenericSurface
from rocketpy.mathutils.vector_matrix import Matrix, Vector
import numpy as np
import math

class Canards(GenericSurface):
    """Defines a canards aerodynamic surface with custom force and moment
    coefficients. The coefficients can be nonlinear functions of the angle of
    attack, sideslip angle, Mach number, Reynolds number, pitch rate, yaw rate
    and roll rate."""
   
    def __init__(self, reference_area, reference_length, coefficients, center_of_pressure, alpha, name="Canards"):
        """
        Create a canards aerodynamic surface, defined by its aerodynamic
        coefficients.

        Important
        ---------
        All the aerodynamic coefficients can be input as callable functions of
        angle of attack, angle of sideslip, Mach number, Reynolds number,
        pitch rate, yaw rate and roll rate. For CSV files, the header must
        contain at least one of the following: "alpha", "beta", "mach",
        "reynolds", "pitch_rate", "yaw_rate" and "roll_rate".

        See Also
        --------
        :ref:`genericsurfaces`.

        Parameters
        ----------
        reference_area : int, float
            Reference area of the aerodynamic surface. Has the unit of meters
            squared. Commonly defined as the rocket's cross-sectional area.
        reference_length : int, float
            Reference length of the aerodynamic surface. Has the unit of meters.
            Commonly defined as the rocket's diameter.
        coefficients: dict
            List of coefficients. If a coefficient is omitted, it is set to 0.
            The valid coefficients are:\n
            cL: str, callable, optional
                Lift coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cQ: str, callable, optional
                Side force coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cD: str, callable, optional
                Drag coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cm: str, callable, optional
                Pitch moment coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cn: str, callable, optional
                Yaw moment coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
            cl: str, callable, optional
                Roll moment coefficient. Can be a path to a CSV file or a callable.
                Default is 0.\n
        center_of_pressure : tuple, list, optional
            The path to a csv file.
        alpha : float
            The angle of attack of the canards in radians.
        name : str, optional
            Name of the aerodynamic surface. Default is 'Canards'.
        """
        
        super().__init__(
            reference_area=reference_area,
            reference_length = reference_length,
            coefficients = coefficients,
            center_of_pressure = center_of_pressure,
            name = name,
        )

        self.alpha = alpha
    
    @property
    def alpha(self):
        """Returns the angle of attack (alpha) in radians of the canards"""
        return self.alpha
    
    @alpha.setter
    def alpha(self, value: float):
        self.alpha = value

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        reynolds,
    ):
        """Computes the forces and moments acting on the aerodynamic surface.
        Used in each time step of the simulation.  This method is valid for
        both linear and nonlinear aerodynamic coefficients.

        Parameters
        ----------
        stream_velocity : tuple of float
            The velocity of the airflow relative to the surface.
        stream_speed : float
            The magnitude of the airflow speed.
        stream_mach : float
            The Mach number of the airflow.
        rho : float
            Air density.
        cp : Vector
            Center of pressure coordinates in the body frame.
        omega: tuple[float, float, float]
            Tuple containing angular velocities around the x, y, z axes.
        reynolds : float
            Reynolds number.
        omega: tuple of float
            Tuple containing angular velocities around the x, y, z axes.

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments
            (pitch, yaw, roll) in the body frame.
        """
        # Stream velocity in standard aerodynamic frame
        stream_velocity = -stream_velocity

        # Angles of attack and sideslip
        alpha = self.alpha
        beta = np.arctan2(stream_velocity[0], stream_velocity[2])

        # Compute aerodynamic forces and moments
        lift, side, drag, pitch, yaw, roll = self._compute_from_coefficients(
            rho,
            stream_speed,
            alpha,
            beta,
            stream_mach,
            reynolds,
            omega[0],  # q
            omega[1],  # r
            omega[2],  # p
        )

        # Conversion from aerodynamic frame to body frame
        rotation_matrix = Matrix(
            [
                [1, 0, 0],
                [0, math.cos(alpha), -math.sin(alpha)],
                [0, math.sin(alpha), math.cos(alpha)],
            ]
        ) @ Matrix(
            [
                [math.cos(beta), 0, -math.sin(beta)],
                [0, 1, 0],
                [math.sin(beta), 0, math.cos(beta)],
            ]
        )
        R1, R2, R3 = rotation_matrix @ Vector([side, -lift, -drag])

        # Dislocation of the aerodynamic application point to CDM
        M1, M2, M3 = Vector([pitch, yaw, roll]) + (cp ^ Vector([R1, R2, R3]))

        return R1, R2, R3, M1, M2, M3