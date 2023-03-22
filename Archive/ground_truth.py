import numpy as np
import matplotlib.pyplot as plt

from config import stripe_spacing_mm, NB_STRIPES, radii_list


def next_pos(r, theta):
    new_pos = np.array([r * np.cos(theta), r * np.sin(-theta)])
    new_pos[0] -= r
    return -new_pos


def create_curvature_vector(s, radius, vector_size=NB_STRIPES):
    positions = []

    if radius == np.inf:
        positions_x = np.linspace(0, s*vector_size-1, vector_size-2)
        positions_y = np.zeros(vector_size-2)
        positions = np.transpose([positions_y, positions_x])

    else:
        theta_rad = s / radius

        for i_stripe in range(vector_size):
            next_position = next_pos(radius, theta_rad * i_stripe)
            positions.append(next_position)

    return positions


def create_curvature():
    fig = plt.figure()
    s = stripe_spacing_mm

    nb_stripes = NB_STRIPES
    curvatures = np.array(radii_list[1:]) * 10 ** -3
    factor = 1

    x = np.linspace(0, (nb_stripes + 1) * s, nb_stripes)
    y = np.zeros(len(x))
    plt.plot(x / factor, y / factor, marker='+', label='Curvature 0 $mm^{-1}$')

    for i_curvature in range(len(curvatures)):
        positions = create_curvature_vector(s, curvatures[i_curvature])

        x, y = np.transpose(positions)
        plt.plot(x / factor, y / factor, marker='+', label='Curvature ' + str(radii_list[i_curvature+1]) + ' $mm^{-1}$')

    plt.legend()
    plt.xlabel('X axis $(cm)$')
    plt.ylabel('Y axis $(cm)$')
    plt.tight_layout()
    plt.axis('equal')
    plt.grid()
    # plt.show()
    return fig




