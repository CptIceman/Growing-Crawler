import pygame


class Segment:
    """Represents a rigid body segment of the crawler.

    A segment encapsulates a Pymunk body and its collision shape.
    It also provides utility methods for rendering and accessing
    physical properties.
    """

    def __init__(
        self,
        body=None,
        shape=None,
        width=40,
        height=10,
        color=(220, 150, 60)
    ):
        """Creates a new crawler segment.

        Args:
            body: The Pymunk body.
            shape: The Pymunk collision shape.
            width: Segment width in pixels.
            height: Segment height in pixels.
            color: RGB color used for rendering.
        """
        self.body = body
        self.shape = shape

        self.width = width
        self.height = height

        self.color = color

    def get_position(self):
        """Returns the segment position.

        Returns:
            pymunk.Vec2d: Segment position.
        """
        return self.body.position

    def get_angle(self):
        """Returns the segment angle.

        Returns:
            float: Segment angle in radians.
        """
        return self.body.angle

    def get_vertices(self):
        """Returns the segment vertices in world coordinates.

        Returns:
            list[pymunk.Vec2d]: The four rotated vertices.
        """
        half_w = self.width / 2
        half_h = self.height / 2

        local_vertices = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]

        return [
            self.body.local_to_world(vertex)
            for vertex in local_vertices
        ]

    def draw(self, screen, screen_height):
        """Draws the segment.

        Args:
            screen: Pygame display surface.
            screen_height: Window height.
        """
        vertices = [
            (vertex.x, screen_height - vertex.y)
            for vertex in self.get_vertices()
        ]

        pygame.draw.polygon(
            screen,
            self.color,
            vertices
        )

    def set_color(self, color):
        """Sets the segment color.

        Args:
            color: RGB tuple.
        """
        self.color = color

    def get_mass(self):
        """Returns the body mass.

        Returns:
            float: Segment mass.
        """
        return self.body.mass

    def get_velocity(self):
        """Returns the body linear velocity.

        Returns:
            pymunk.Vec2d: Linear velocity.
        """
        return self.body.velocity

    def get_angular_velocity(self):
        """Returns the body angular velocity.

        Returns:
            float: Angular velocity.
        """
        return self.body.angular_velocity