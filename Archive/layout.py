import PySimpleGUI as sg
from config import phantom_materials, rail_materials, radii_list, pattern_type, pattern_colour, pattern_frequency, EXPERIMENT_ARGS, EXPERIMENT_ARGS_TO_KEY
from plotting import draw_figure, plot_curvature_ground_truth
from Archive.ground_truth import create_curvature
from animations import generate_args
# VARS CONSTS:
_VARS = {'window': False}

def settings_button(image_subsample=1):
    settings_button = sg.Button('', key='-SETTINGS-',
                                button_color=(sg.theme_background_color(), sg.theme_background_color()),
                                border_width=0, image_subsample=image_subsample)
    return settings_button


def create_parameters_column_list():
    label_column_list = [[sg.Text('Phantom material: ')], [sg.Text('Rail material: ')], [sg.Text('Pattern Type: ')],
                         [sg.Text('Pattern colour: ')], [sg.Text('Pattern frequency :')],
                         [sg.Text('Curvature (1/mm) :')]]

    input_combo_column_list = [
        [sg.InputCombo(phantom_materials, default_value=phantom_materials[0],
                       enable_events=True, size=(7,1),
                       key = '-PHANTOM-')],
        [sg.InputCombo(rail_materials, default_value=rail_materials[0],
                       enable_events=True, size=(7, 1),
                       key='-RAIL-')],
        [sg.InputCombo(pattern_type, default_value=pattern_type[0],
                       enable_events=True, size=(7, 1),
                       key='-PATTERN-')],
        [sg.InputCombo(pattern_colour, default_value=pattern_colour[0],
                       enable_events=True, size=(7, 1),
                       key='-COLOUR-')],
        [sg.InputCombo(pattern_frequency, default_value=pattern_frequency[0],
                       enable_events=True, size=(7, 1),
                       key='-FREQUENCY-')],
        [sg.InputCombo(radii_list, default_value=radii_list[0],
                       enable_events=True, size=(7, 1),
                       key='-RADIUS-')]]
    parameters_column_list = [[
        sg.Frame('Parameters', [[sg.Column(label_column_list, element_justification='l'),
                                 sg.Column(input_combo_column_list, element_justification='c')]]
                 )]]
    return parameters_column_list

def generate_recording_layout():
    # sg.theme('LightGreen5')
    recording_layout = [sg.Frame('Record',
                                 [[
                                     sg.Button(
                                         image_data=play,
                                         button_color=(sg.theme_background_color(), sg.theme_background_color()),
                                         border_width=0, key='-START_REC-'),
                                     sg.Button(
                                         image_data=stop,
                                         button_color=(sg.theme_background_color(), sg.theme_background_color()),
                                         border_width=0, key='-STOP_REC-')],
                                     [sg.Text('Elapsed time :'), sg.Text('', key='-TIME-', size=(8, 1))]]
                                 )]
    return recording_layout

def generate_parameters_column():
    recording_frame = [sg.Frame('Experiment Folders',
                                [[sg.Text('Load experiment ')],
                                [sg.InputCombo(tuple(), default_value='./input_images/',size=(25, 1), key='-EXP_FOLDER-', enable_events=True)],
                                [sg.Text('Select save folder  ')],
                                [sg.InputCombo(tuple(), default_value='./output_trajectories', size=(25, 1), key='-SAVE_FOLDER-', enable_events=True)]])]

    layout = [
            [sg.Text('Record player', font='Any 20')],
            recording_frame]

    parameters_column = create_parameters_column_list()
    layout.extend(parameters_column)

    return sg.Column(layout)


# fig = create_curvature()
layout = [[generate_parameters_column(),
           sg.Canvas(size = (640, 480),key='-CANVAS-')]]
window = sg.Window('test', layout, finalize=True)


canvas_elem = window['-CANVAS-']
canvas = canvas_elem.TKCanvas
# Generate the argument to draw the initial plot
animation_args, animation_kwargs = generate_args()
fig, axs = animation_args
# Draw the figure for the first time
fig_agg = draw_figure(canvas, fig)

experiment_args = dict.fromkeys(EXPERIMENT_ARGS)
event, values = window.read(timeout=1)
experiment_args.update({'Number of stripes': values['-FREQUENCY-']})

while True:
    event, values = window.read()

    if event in (sg.WIN_CLOSED, 'Exit'):
        break

        #plot corresponding graph
    fig_agg.draw()

    if event == '-PHANTOM-':
        experiment_args['phantom_material'] = values['-PHANTOM-']

    if event == '-RAIL-':
        experiment_args['rail_material'] = values['-RAIL-']

    if event == '-RADIUS-':
        # Here do we need to have two dictionary? Why are they overlapping?
        radius_mm = values['-RADIUS-']
        experiment_args['compared_curvature'] = radius_mm

        # Update the compared curvature when the radius is changed
        create_curvature()
        plot_curvature_ground_truth(axs, radius_mm, 20)

    # window.close()




# while True:
#    event, values = window.read()
#    print (event, values)
#    if event in (sg.WIN_CLOSED, 'Exit'):
#       break
#
# window.close()

