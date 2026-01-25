import warnings

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.plots.aero_surface_plots import _AeroSurfacePlots
from rocketpy.prints.aero_surface_prints import _AeroSurfacePrints

from .aero_surface import AeroSurface


class Canards(AeroSurface):
    """Canards class. Inherits from AeroSurface.

    This class represents canard aerodynamic surfaces that can generate both
    lift forces and moments. Unlike the AirBrakes class, canards account for
    lift coefficients and center of pressure variations based on angle of
    attack and Mach number.

    Attributes
    ----------
    Canards.drag_coefficient : Function
        Drag coefficient as a function of angle of attack and Mach number.
    Canards.lift_coefficient_curve : int, float, callable, array, string, Function
        Curve that defines the lift coefficient as a function of angle of
        attack and Mach number. Used as the source of `Canards.lift_coefficient`.
    Canards.center_of_pressure_curve : int, float, callable, array, string, Function
        Curve that defines the center of pressure position (in local coordinates)
        as a function of angle of attack and Mach number.
    Canards.reference_area : int, float
        Reference area used to calculate the lift and drag forces.
        Units of m^2.
    Canards.reference_length : int, float
        Reference length used to calculate pitching moments. Units of m.
    Canards.deployment_level : float
        Current deployment level, ranging from 0 to 1. Deployment level is the
        fraction of the total canard area that is deployed.
    Canards.clamp : bool, optional
        If True, the deployment level will be clamped to 0 or 1 if out of bounds.
        If False, a warning will be raised. Default is True.
    Canards.name : str
        Name of the canards.
    Canards.angular_position : float
        Angular position of the canards around the rocket's longitudinal axis
        in radians, measured from the x-axis in the body frame. Used to calculate
        roll moments from off-axis lift forces. Default is 0 (on the positive x-axis).
    """

    def __init__(
        self,
        lift_coefficient_curve,
        center_of_pressure_curve,
        reference_area,
        reference_length,
        drag_coefficient_curve=None,
        clamp=True,
        deployment_level=0,
        angular_position=0,
        name="Canards",
    ):
        """Initializes the Canards class.

        Parameters
        ----------
        lift_coefficient_curve : int, float, callable, array, string, Function
            This parameter represents the lift coefficient associated with the
            canards.

            - If a constant, it should be an integer or a float representing a
              fixed lift coefficient value.
            - If a function, it must take two parameters: angle of attack (in
              radians) and Mach number, and return the lift coefficient.
            - If an array, it should be a 2D array with three columns: the first
              column for angle of attack (radians), the second for Mach number,
              and the third for the corresponding lift coefficient.
            - If a string, it should be the path to a .csv or .txt file. The
              file must contain three columns: angle of attack (radians), Mach
              number, and lift coefficient.
            - If a Function, it must take two parameters: angle of attack and
              Mach number, and return the lift coefficient.

        center_of_pressure_curve : int, float, callable, array, string, Function
            This parameter represents the center of pressure position. The format
            is similar to lift_coefficient_curve, but it should provide positions
            along the rocket's axis (typically x-position in local coordinates).

            - If a constant, it should represent a fixed axial position (m).
            - If a function, it must take two parameters: angle of attack and
              Mach number, and return the center of pressure position.
            - If an array, it should be a 2D array with three columns.
            - If a string, it should be the path to a .csv or .txt file.
            - If a Function, it must take two parameters: angle of attack and
              Mach number.

        reference_area : int, float
            Reference area used to calculate the lift force of the canards
            from the lift coefficient curve. Units of m^2.

        reference_length : int, float
            Reference length used to calculate pitching moments from lift forces.
            Typically the rocket diameter or a characteristic length. Units of m.

        drag_coefficient_curve : int, float, callable, array, string, Function, optional
            Drag coefficient curve with the same format as lift_coefficient_curve.
            If None, no drag is calculated. Default is None.

        clamp : bool, optional
            If True, the deployment level will be clamped to 0 or 1 if out of
            bounds. If False, a warning will be raised. Default is True.

        deployment_level : float, optional
            Initial deployment level, ranging from 0 to 1. Deployment level is
            the fraction of the total canard area that is deployed. Default is 0.

        angular_position : float, optional
            Angular position of the canards around the rocket's longitudinal axis
            in radians, measured from the x-axis in the body frame. This is used
            to calculate the radial moment arm for roll moment generation. For
            example, 0 rad places the canard at x-axis, π/2 at y-axis, etc.
            Default is 0.

        name : str, optional
            Name of the canards. Default is "Canards".

        Returns
        -------
        None
        """
        super().__init__(name, reference_area, reference_length)
        
        # Store curves
        self.lift_coefficient_curve = lift_coefficient_curve
        self.center_of_pressure_curve = center_of_pressure_curve
        self.drag_coefficient_curve = drag_coefficient_curve
        
        # Create Function objects for lift coefficient
        self.cl = Function(
            lift_coefficient_curve,
            inputs=["Angle of Attack (rad)", "Mach"],
            outputs="Lift Coefficient",
        )
        
        # Create Function object for center of pressure
        self.cp_function = Function(
            center_of_pressure_curve,
            inputs=["Angle of Attack (rad)", "Mach"],
            outputs="Center of Pressure (m)",
        )
        
        # Create Function object for drag coefficient if provided
        if drag_coefficient_curve is not None:
            self.cd = Function(
                drag_coefficient_curve,
                inputs=["Angle of Attack (rad)", "Mach"],
                outputs="Drag Coefficient",
            )
        else:
            self.cd = Function(
                lambda alpha, mach: 0,
                inputs=["Angle of Attack (rad)", "Mach"],
                outputs="Drag Coefficient",
            )
        
        # Lift coefficient alpha derivative (for moment calculations)
        self.clalpha = Function(
            lambda mach: 0,  # This will be evaluated from the lift coefficient curve
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        
        # Deployment level control
        self.clamp = clamp
        self.initial_deployment_level = deployment_level
        self.deployment_level = deployment_level
        
        # Angular position for roll moment calculation
        self.angular_position = angular_position
        
        # Center of pressure in local coordinates
        self.cpx = 0
        self.cpy = 0
        self.cpz = 0
        self.cp = (self.cpx, self.cpy, self.cpz)

    @property
    def deployment_level(self):
        """Returns the deployment level of the canards."""
        return self._deployment_level

    @deployment_level.setter
    def deployment_level(self, value):
        """Sets the deployment level of the canards with bounds checking."""
        if value < 0 or value > 1:
            if self.clamp:
                value = np.clip(value, 0, 1)
            else:
                warnings.warn(
                    f"Deployment level of {self.name} is smaller than 0 or "
                    + "larger than 1. Extrapolation for the lift and center of "
                    + "pressure curves will be used.",
                    UserWarning,
                )
        self._deployment_level = value

    def _reset(self):
        """Resets the canards to their initial state. This is run at the
        beginning of each simulation to ensure the canards are in the correct
        state."""
        self.deployment_level = self.initial_deployment_level

    def evaluate_center_of_pressure(self, alpha=0, mach=0):
        """Evaluates the center of pressure of the canards in local coordinates.

        Parameters
        ----------
        alpha : float, optional
            Angle of attack in radians. Default is 0.
        mach : float, optional
            Mach number. Default is 0.

        Returns
        -------
        None
        """
        # Get center of pressure position from the curve
        cp_position = self.cp_function.get_value_opt(alpha, mach)
        
        # For canards, the center of pressure is typically along the rocket axis
        # Position the CP at the computed location along the z-axis (rocket's centerline)
        self.cpx = 0  # No lateral offset in x direction
        self.cpy = 0  # No lateral offset in y direction
        self.cpz = cp_position  # Axial position along z-axis (nose direction)
        self.cp = (self.cpx, self.cpy, self.cpz)

    def evaluate_lift_coefficient(self):
        """Evaluates the lift coefficient curve of the canards.

        The lift coefficient is already stored as a Function object during
        initialization. This method ensures compatibility with the AeroSurface
        interface.

        Returns
        -------
        None
        """
        # The lift coefficient is already set in __init__
        # This method is here for interface compatibility
        pass

    def evaluate_geometrical_parameters(self):
        """Evaluates the geometrical parameters of the canards.

        Returns
        -------
        None
        """
        # Can be extended to compute additional geometric properties
        pass

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        *args,
    ):
        """Computes the forces and moments acting on the canards.

        This method overrides the parent class method to account for lift forces
        and the varying center of pressure based on angle of attack.

        Parameters
        ----------
        stream_velocity : tuple
            Tuple containing the stream velocity components in the body frame
            (vx, vy, vz).
        stream_speed : float
            Speed of the stream in m/s.
        stream_mach : float
            Mach number of the stream.
        rho : float
            Density of the stream in kg/m^3.
        cp : tuple
            Reference center of pressure coordinates in the body frame.
        args : tuple
            Additional arguments.

        Returns
        -------
        tuple of float
            The aerodynamic forces (R1, R2, R3) and moments (M1, M2, M3)
            in the body frame. Forces are in Newtons, moments in Newton-meters.
        """
        R1, R2, R3, M1, M2, M3 = 0, 0, 0, 0, 0, 0
        
        # Apply deployment level to reference area
        effective_area = self.reference_area * self.deployment_level
        
        # Extract velocity components
        stream_vx, stream_vy, stream_vz = stream_velocity
        
        # Calculate lateral velocity magnitude (perpendicular to rocket axis)
        lateral_velocity_magnitude = np.sqrt(stream_vx**2 + stream_vy**2)
        
        if lateral_velocity_magnitude > 1e-6:  # Avoid division by zero
            # Normalize component stream velocity in body frame
            stream_vzn = stream_vz / stream_speed
            
            # Calculate angle of attack
            if -1 * stream_vzn <= 1:
                attack_angle = np.arccos(-stream_vzn)
                
                # Get lift and drag coefficients
                c_lift = self.cl.get_value_opt(attack_angle, stream_mach)
                c_drag = self.cd.get_value_opt(attack_angle, stream_mach)
                
                # Calculate lift force magnitude
                q_bar = 0.5 * rho * stream_speed**2  # Dynamic pressure
                lift = q_bar * effective_area * c_lift
                drag = q_bar * effective_area * c_drag
                
                # Lift force components (perpendicular to velocity)
                lift_xb = lift * (stream_vx / lateral_velocity_magnitude)
                lift_yb = lift * (stream_vy / lateral_velocity_magnitude)
                
                # Drag force components (opposite to velocity direction)
                if stream_speed > 1e-6:
                    drag_xb = -drag * (stream_vx / stream_speed)
                    drag_yb = -drag * (stream_vy / stream_speed)
                    drag_zb = -drag * (stream_vz / stream_speed)
                else:
                    drag_xb, drag_yb, drag_zb = 0, 0, 0
                
                # Total forces
                R1 = lift_xb + drag_xb
                R2 = lift_yb + drag_yb
                R3 = drag_zb
                
                # Get center of pressure for moment calculation
                self.evaluate_center_of_pressure(attack_angle, stream_mach)
                cpz = self.cpz if hasattr(self, 'cpz') else 0
                
                # Calculate moments about the center of mass
                # M = r × F (cross product of position and force)
                # Radial position vector: r = [R*cos(θ), R*sin(θ), cpz]
                # where R is reference_length (rocket radius) and θ is angular_position
                radial_distance = self.reference_length
                cos_angle = np.cos(self.angular_position)
                sin_angle = np.sin(self.angular_position)
                
                # Pitch moment: from axial offset (cpz) and yaw lift
                M1 = -cpz * lift_yb
                # Yaw moment: from axial offset (cpz) and pitch lift
                M2 = cpz * lift_xb
                # Roll moment: from radial offset and lateral lift forces
                # M3 = r_x * F_y - r_y * F_x
                M3 = radial_distance * (cos_angle * lift_yb - sin_angle * lift_xb)
        
        return R1, R2, R3, M1, M2, M3

    def info(self):
        """Prints summarized information of the canards.

        Returns
        -------
        None
        """
        print(f"Canards: {self.name}")
        print(f"  Reference Area: {self.reference_area:.4f} m²")
        print(f"  Reference Length: {self.reference_length:.4f} m")
        print(f"  Angular Position: {np.degrees(self.angular_position):.1f}°")
        print(f"  Deployment Level: {self.deployment_level:.2f}")

    def all_info(self):
        """Prints all information of the canards.

        Returns
        -------
        None
        """
        self.info()

    def to_dict(self, **kwargs):
        """Converts the canards object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the canards object.
        """
        return {
            "lift_coefficient_curve": self.lift_coefficient_curve,
            "center_of_pressure_curve": self.center_of_pressure_curve,
            "drag_coefficient_curve": self.drag_coefficient_curve,
            "reference_area": self.reference_area,
            "reference_length": self.reference_length,
            "clamp": self.clamp,
            "deployment_level": self.initial_deployment_level,
            "angular_position": self.angular_position,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        """Creates a canards object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with canards properties.

        Returns
        -------
        Canards
            Canards object created from the dictionary.
        """
        return cls(
            lift_coefficient_curve=data.get("lift_coefficient_curve"),
            center_of_pressure_curve=data.get("center_of_pressure_curve"),
            reference_area=data.get("reference_area"),
            reference_length=data.get("reference_length"),
            drag_coefficient_curve=data.get("drag_coefficient_curve"),
            clamp=data.get("clamp", True),
            deployment_level=data.get("deployment_level", 0),
            angular_position=data.get("angular_position", 0),
            name=data.get("name", "Canards"),
        )
