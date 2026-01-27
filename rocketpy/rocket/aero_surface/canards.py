from .generic_surface import GenericSurface

class Canards(GenericSurface):
   """Defines a canards aerodynamic surface with custom force and moment
    coefficients. The coefficients can be nonlinear functions of the angle of
    attack, sideslip angle, Mach number, Reynolds number, pitch rate, yaw rate
    and roll rate."""
   
   def __init__(
        self,
        reference_area,
        reference_length,
        coefficients,
        center_of_pressure,
        name="Canards",
    ):
        """Create a canards aerodynamic surface, defined by its aerodynamic
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