import PySimpleGUI as sg
from Archive.take_snapshot_images import start_snapshot_capture

RADII_MM = [30, 50, 70, 90, 110]

def start_snapshot_gui():
    sg.theme('LightGreen5')

    layout = [[ sg.Text('Set snapshot interval (s):'), sg.InputText(size=(5,1), enable_events=True, key='-SECONDS-'),
              sg.Text('Set number of images:'), sg.InputText(size=(5,1), enable_events=True, key='-NUMBER-'),

                sg.Button("Start Image Capture")]]
    window = sg.Window('test_window', layout)

    #create event loop
    while True:
        event, values = window.read()
        text_input = values['-SECONDS-']
        text_input_two = values['-NUMBER-']

        if event == "Start Image Capture":
            sg.popup('Starting Image Capture of ' + str(text_input_two) + 'with ' + str(text_input) + ' second intervals')
            start_snapshot_capture(text_input, text_input_two)
            sg.popup('Images saved')
        elif event == sg.WIN_CLOSED:
            break

    window.close()

start_snapshot_gui()








