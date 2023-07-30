import PySimpleGUI as sg
import cv2

RADII_MM = [None, '50mm', '70mm', '90mm', '110mm']
COLOUR_OPTIONS = ['yellow', 'red', 'orange']

def start_snapshot_capture(interval, num_images, colour, radius, save_path):
    # Convert interval to integer and set default radius to None
    interval = int(interval)
    radius = None if radius == '' else radius

    # Set up camera
    cam = cv2.VideoCapture(0)

    # Capture images
    for i in range(int(num_images)):
        ret, frame = cam.read()

        # Save image with filename in the format: colour_radius_itern.jpg
        filename = f"{colour}_{radius}_iter{i+1}.jpg"
        filepath = os.path.join(save_path, filename)
        cv2.imwrite(filepath, frame)

        # Wait for specified interval before capturing next image
        cv2.waitKey(interval * 1000)

    # Release camera
    cam.release()



def start_snapshot_gui():
    sg.theme('LightGreen5')

    layout = [[ sg.Text('Set snapshot interval (s):'), sg.InputText(size=(5,1), enable_events=True, key='-SECONDS-'),
              sg.Text('Set number of images:'), sg.InputText(size=(5,1), enable_events=True, key='-NUMBER-'),
              sg.Text('Select colour:'), sg.Combo(COLOUR_OPTIONS, size=(10, 1), key='-COLOUR-'),
              sg.Text('Select radius:'), sg.Combo(RADII_MM, size=(10, 1), key='-RADIUS-'),
              sg.Text('Save images to:'), sg.InputText(key='-SAVE_PATH-'), sg.FolderBrowse(),

              sg.Button("Start Image Capture")]]
    window = sg.Window('test_window', layout)

    #create event loop
    while True:
        event, values = window.read()
        text_input = values['-SECONDS-']
        text_input_two = values['-NUMBER-']
        colour = values['-COLOUR-']
        radius = values['-RADIUS-']
        save_path = values['-SAVE_PATH-']

        if event == "Start Image Capture":
            sg.popup('Starting Image Capture of ' + str(text_input_two) + ' with ' + str(text_input) + ' second intervals')
            start_snapshot_capture(text_input, text_input_two, colour, radius, save_path)
            sg.popup('Images saved')
        elif event == sg.WIN_CLOSED:
            break

    window.close()


start_snapshot_gui()















