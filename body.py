import pymunk

from joint import Joint
from segment import Segment

class Body:
    def __init__(self, gravity=(0, -900), num_segments=6):
        self.space = pymunk.Space()
        self.space.gravity = gravity

        self.segments = []
        self.joints = []

        self._create_ground()
        
        self._create_worm(num_segments=num_segments)

    def _create_ground(self):
        ground = pymunk.Segment(self.space.static_body, (-1000, 50), (1000, 50), 5)
        ground.friction = 1.0
        ground.elasticity = 0.1
        self.space.add(ground)

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
            shape.friction = 0.4
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
            joint.set_limits(-0.8, 0.8)
            joint.initialize(self.space)
            self.joints.append(joint)

    def step(self, dt=1 / 60.0):
        self.space.step(dt)

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