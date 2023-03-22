import PySimpleGUI as sg

start_button = sg.Button(key="-START-",
                         button_color=(sg.theme_background_color(), sg.theme_background_color()),
                         image_filename='icons/start.png', image_size=(150, 150), image_subsample=8,
                         border_width=0)

stop_button = sg.Button(key="-STOP-",
                        button_color=(sg.theme_background_color(), sg.theme_background_color()),
                        image_filename='icons/stop.png', image_size=(150, 150), image_subsample=8,
                        border_width=0)