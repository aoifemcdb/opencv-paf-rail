"""Contains all common code for defining and testing color boundaries"""


class ColorBoundaries:
    def get_color_boundaries(self, color):
        # Define the lower and upper color bounds for the specified color in hsv colorspace
        if color == 'red': #this is actually red
            lower_color = (0, 128, 64)
            upper_color = (150, 255, 255)
            lower_color2 = (170, 50, 50)
            upper_color2 = (180, 255, 255)
        elif color == 'blue': #this is actually blue - temporarily changing to run auto
            lower_color = (0, 128, 128)
            upper_color = (120, 255, 255)
        elif color == 'green': #this is actually green
            lower_color = (35, 50, 50)
            upper_color = (90, 255, 255)
        elif color == 'yellow':
            lower_color = (15, 100, 100)
            upper_color = (35, 255, 255)
        elif color == 'orange':
            lower_color = (10, 128, 128)
            upper_color = (20, 255, 255)
        else:
            raise ValueError("Invalid color specified. Must be 'red', 'blue', 'green', 'yellow', or 'orange'.")

        # Convert the color bounds to numerical tuples
        lower_color_tuple = tuple(lower_color)
        upper_color_tuple = tuple(upper_color)

        # Return the numerical tuples
        return lower_color_tuple, upper_color_tuple
