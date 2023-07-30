import numpy as np
from scipy.optimize import leastsq
from scipy.spatial import Delaunay
from scipy.optimize import curve_fit
from scipy.optimize import leastsq

class CurvatureCalculator:
    @staticmethod
    def localized_curve_fitting(curved_downsampled, segment_length):
        num_points = curved_downsampled.shape[0]
        num_segments = int(np.ceil(num_points / segment_length))

        curvatures = []

        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length

            # Ensure the end index is within the data range
            if end_idx > num_points:
                end_idx = num_points

            segment = curved_downsampled[start_idx:end_idx, :]

            segment_x = segment[:, 0]
            segment_y = segment[:, 1]

            # Perform your circle fitting method here
            # ...
            # For example, using the circle_fitting_method function

            def circle_fitting_method(x, y):
                # Fit a circle to the points using least squares optimization

                def fit_func(params, x, y):
                    xc, yc, r = params
                    return (x - xc) ** 2 + (y - yc) ** 2 - r ** 2

                def residuals(params, x, y):
                    return fit_func(params, x, y)

                # Use the mean of the x and y coordinates as initial guesses for the circle center
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                initial_guess = [x_mean, y_mean, 1.0]

                circle_center, _ = leastsq(residuals, initial_guess, args=(x, y))
                xc, yc, r = circle_center

                # Calculate the curvature as the reciprocal of the circle radius
                curvature = 1.0 / r

                return curvature

            curvature = circle_fitting_method(segment_x, segment_y)

            curvatures.append(curvature)

        average_curvature = np.mean(curvatures)

        return average_curvature

    @staticmethod
    def derivatives_method(points):
        # Calculate the first and second derivatives of the x and y coordinates
        x = points[:, 0]
        y = points[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Calculate the curvature using the formula:
        # curvature = |dx*ddy - ddx*dy| / (dx^2 + dy^2)^(3/2)
        numerator = np.abs(dx * ddy - ddx * dy)
        denominator = (dx ** 2 + dy ** 2) ** (3/2)
        curvature = numerator / denominator
        curvature = np.average(curvature)

        # Return the curvature as a NumPy array
        return curvature

    @staticmethod
    def circle_fitting_method(points):
        # Fit a circle to the points using least squares optimization
        x = points[:, 0]
        y = points[:, 1]

        def fit_func(params, x, y):
            xc, yc, r = params
            return (x - xc)**2 + (y - yc)**2 - r**2

        def residuals(params, x, y):
            return fit_func(params, x, y)

        # Use the mean of the x and y coordinates as initial guesses for the circle center
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        initial_guess = [x_mean, y_mean, 1.0]

        circle_center, _ = leastsq(residuals, initial_guess, args=(x, y))
        xc, yc, r = circle_center

        # Calculate the curvature as the reciprocal of the circle radius
        curvature = 1.0 / r

        # Return the curvature as a NumPy array
        return curvature, circle_center

    @staticmethod
    def finite_difference_method(points):
        # Calculate the first and second derivatives of the x and y coordinates
        x = points[:, 0]
        y = points[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Calculate the curvature using the finite difference method:
        # curvature = |dx*ddy - ddx*dy| / (dx^2 + dy^2)^(3/2)
        numerator = np.abs(dx * ddy - ddx * dy)
        denominator = (dx ** 2 + dy ** 2) ** (3/2)
        curvature = numerator / denominator
        curvature = np.average(curvature)

        # Return the curvature as a NumPy array
        return curvature

    @staticmethod
    def gaussian_curvature_method(points):
        # Construct the Delaunay triangulation of the points
        tri = Delaunay(points)

        # Calculate the areas of the triangles
        areas = []
        for i in range(tri.nsimplex):
            simplex = tri.simplices[i]
            x = points[simplex][:, 0]
            y = points[simplex][:, 1]
            a = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
            b = np.sqrt((x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2)
            c = np.sqrt((x[2] - x[0]) ** 2 + (y[2] - y[0]) ** 2)
            s = (a + b + c) / 2
            areas.append(np.sqrt(s * (s - a) * (s - b) * (s - c)))

        # Calculate the angles of the triangles
        angles = []
        for i in range(tri.nsimplex):
            simplex = tri.simplices[i]
            x = points[simplex][:, 0]
            y = points[simplex][:, 1]
            a = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
            b = np.sqrt((x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2)
            c = np.sqrt((x[2] - x[0]) ** 2 + (y[2] - y[0]) ** 2)
            angles.append(np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))

        # Calculate the Gaussian curvature using the formula:
        # K = (2*pi - sum of angles) / sum of areas
        curvature = (2 * np.pi - np.sum(angles)) / np.sum(areas)

        # Return the Gaussian curvature
        return curvature

    @staticmethod
    def delaunay_triangulation_method(points):
        # Create a Delaunay triangulation object from the set of 2D coordinates
        triangulation = Delaunay(points)

        # Calculate the area of each triangle in the triangulation
        areas = np.zeros(triangulation.nsimplex)
        for i, simplex in enumerate(triangulation.simplices):
            p1, p2, p3 = triangulation.points[simplex]
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p1 - p3)
            s = (a + b + c) / 2.0
            areas[i] = np.sqrt(s * (s - a) * (s - b) * (s - c))

        # Calculate the curvature using the formula:
        # curvature = (2 * pi) / sum(areas) for each vertex in the triangulation
        curvatures = np.zeros(triangulation.npoints)
        for i, vertex in enumerate(triangulation.vertices):
            triangle_indices = np.where(np.any(triangulation.simplices == vertex, axis=1))[0]
            non_degenerate_triangles = np.where(areas[triangle_indices] > 0)[0]
            if len(non_degenerate_triangles) > 0:
                curvatures[i] = (2 * np.pi) / np.sum(areas[triangle_indices][non_degenerate_triangles])
            else:
                curvatures[i] = np.nan

        # Return the curvatures as a NumPy array
        curvatures = np.average(curvatures)
        return curvatures
