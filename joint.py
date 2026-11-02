import pymunk


class Joint:
    """Represents an articulation between two crawler segments.

    A joint encapsulates all Pymunk constraints required to connect two
    rigid bodies, including the pivot joint, motor and optional rotation
    limits.
    """

    def __init__(
        self,
        body_a=None,
        body_b=None,
        anchor_a=(0, 0),
        anchor_b=(0, 0)
    ):
        """Creates a new joint.

        Args:
            body_a: First body connected by the joint.
            body_b: Second body connected by the joint.
            anchor_a: Local anchor point on the first body.
            anchor_b: Local anchor point on the second body.
        """
        self.body_a = body_a
        self.body_b = body_b

        self.anchor_a = anchor_a
        self.anchor_b = anchor_b

        self.space = None

        self.pivot = None
        self.motor = None
        self.limit = None

        self.default_motor_speed = 0.0
        self.max_motor_force = 10000

        self.min_angle = None
        self.max_angle = None

        self.initialized = False

        

    def initialize(self, space):
        """Initializes the joint.

        Creates the internal Pymunk constraints and adds them to the
        specified physics space.

         Args:
         space: The Pymunk physics space.
 
         Raises:
         ValueError: If one or both bodies have not been assigned.
        """
        
        if self.initialized:
            return

        if self.body_a is None or self.body_b is None:
            raise ValueError("The bodies have not been assigned.")

        self.space = space

        self.pivot = pymunk.PivotJoint(
            self.body_a,
            self.body_b,
            self.anchor_a,
            self.anchor_b
        )
        self.pivot.collide_bodies = False

        self.motor = pymunk.SimpleMotor(
            self.body_a,
            self.body_b,
            self.default_motor_speed
        )

        self.motor.max_force = self.max_motor_force

        if self.min_angle is not None and self.max_angle is not None:
            self.limit = pymunk.RotaryLimitJoint(
                self.body_a,
                self.body_b,
                self.min_angle,
                self.max_angle
            )

        self.space.add(self.pivot)
        self.space.add(self.motor)

        if self.limit is not None:
            self.space.add(self.limit)

        self.initialized = True

    def remove(self):
        if not self.initialized:
            return

        self.space.remove(self.pivot)
        self.space.remove(self.motor)

        if self.limit is not None:
            self.space.remove(self.limit)

        self.initialized = False

    def set_motor_speed(self, speed):
        self.default_motor_speed = speed

        if self.motor is not None:
            self.motor.rate = speed

    def stop_motor(self):
        self.set_motor_speed(0.0)

    def set_motor_force(self, force):
        self.max_motor_force = force

        if self.motor is not None:
            self.motor.max_force = force

    def set_limits(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def get_relative_angle(self):
        if self.body_a is None or self.body_b is None:
            return 0.0

        return self.body_b.angle - self.body_a.angle