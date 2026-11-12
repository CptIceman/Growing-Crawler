import pymunk

from joint import Joint
from segment import Segment

# Ground runs from -GROUND_HALF_LENGTH to +GROUND_HALF_LENGTH. A good gait covers
# ~800 units per 500-step episode, so this must stay far beyond reach.
GROUND_HALF_LENGTH = 100_000

# Inchworm body parameters.
SEGMENT_FRICTION = 0.2   # low: forward sliding is easy, the bristles provide the grip
JOINT_LIMIT = 0.6        # max bend per joint (rad), so the body can't fold sharply
MAX_TOTAL_BEND = 1.0     # max head-to-tail relative angle (rad); a ball needs ~2.5
# Bristles (setae): fraction of backward velocity removed from grounded segments
# after each physics step. 1.0 = perfect ratchet, 0.0 = no bristles.
BRISTLE_GRIP = 1.0

class Body:
    def __init__(self, gravity=(0, -900), num_segments=6):
        self.space = pymunk.Space()
        self.space.gravity = gravity

        self.segments = []
        self.joints = []

        self._create_ground()
        
        self._create_worm(num_segments=num_segments)

    def _create_ground(self):
        ground = pymunk.Segment(
            self.space.static_body, (-GROUND_HALF_LENGTH, 50), (GROUND_HALF_LENGTH, 50), 5
        )
        ground.friction = 1.0
        ground.elasticity = 0.1
        self.space.add(ground)
        self.ground = ground

    def _create_worm(self, num_segments, start_pos=(200, 300), width=40, height=10, mass=1.0):
        """
        Dynamically constructs a worm body consisting of chained segments 
        interconnected by pivot and constraint joints.
        """
        x_start, y_start = start_pos
        moment = pymunk.moment_for_box(mass, (width, height))

        
        for i in range(num_segments):
            pos_x = x_start + (i * width)
            pos_y = y_start

            body = pymunk.Body(mass, moment)
            body.position = (pos_x, pos_y)

            shape = pymunk.Poly.create_box(body, (width, height))
            shape.friction = SEGMENT_FRICTION
            shape.elasticity = 0.2
            shape.filter = pymunk.ShapeFilter(group=1)

            self.space.add(body, shape)
            
            self.segments.append(Segment(body, shape, width, height))

        
        for i in range(num_segments - 1):
            segment_a = self.segments[i]
            segment_b = self.segments[i + 1]

            anchor_a = (width / 2.0, 0.0)
            anchor_b = (-width / 2.0, 0.0)


            joint = Joint(
                body_a=segment_a.body,  
                body_b=segment_b.body,
                anchor_a=anchor_a,
                anchor_b=anchor_b
            )
            joint.set_limits(-JOINT_LIMIT, JOINT_LIMIT)
            joint.initialize(self.space)
            self.joints.append(joint)

        # Ball blocker: the total bend (sum of joint angles) equals the tail's angle
        # minus the head's, so limiting head vs tail stops the body curling up while
        # still allowing an arch (which bends up and back down, summing to ~0).
        self.bend_limit = pymunk.RotaryLimitJoint(
            self.segments[0].body, self.segments[-1].body, -MAX_TOTAL_BEND, MAX_TOTAL_BEND
        )
        self.space.add(self.bend_limit)

    def step(self, dt=1 / 60.0):
        self.space.step(dt)
        self._apply_bristles()

    def is_grounded(self, segment):
        """Returns True if the segment is touching the ground this step."""
        touching = False

        def check(arbiter):
            nonlocal touching
            if self.ground in arbiter.shapes:
                touching = True

        segment.body.each_arbiter(check)
        return touching

    def ground_contacts(self):
        return [self.is_grounded(segment) for segment in self.segments]

    def _apply_bristles(self):
        """One-way grip: grounded segments can slide forward but not backward.

        pymunk only has direction-independent friction, so this edits velocity
        directly, which is a common game-physics shortcut, not real contact physics.
        """
        for segment in self.segments:
            vx, vy = segment.body.velocity
            if vx < 0 and self.is_grounded(segment):
                segment.body.velocity = (vx * (1.0 - BRISTLE_GRIP), vy)

    def get_segment_positions(self):
        return [
            {"pos": segment.get_position(), "angle": segment.get_angle(),
             "w": segment.width, "h": segment.height}
            for segment in self.segments
        ]

    def center_of_mass(self):
        total_mass = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        for segment in self.segments:
            m = segment.get_mass()
            pos = segment.get_position()
            weighted_x += pos.x * m
            weighted_y += pos.y * m
            total_mass += m
        if total_mass == 0:
            return (0.0, 0.0)
        return (weighted_x / total_mass, weighted_y / total_mass)
    
    def apply_action(self, action):
        """Applies the agent actions to all joints.

        Args:
            action: Sequence containing one action per joint.

        Raises:
            ValueError: If the number of actions does not match the number of joints.
        """
        if len(action) != len(self.joints):
            raise ValueError(
                "Action length must match the number of joints."
            )

        for joint, speed in zip(self.joints, action):
            joint.set_motor_speed(speed)