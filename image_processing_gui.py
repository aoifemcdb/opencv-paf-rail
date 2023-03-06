import PySimpleGUI as sg
from thresholding import threshold_red
import io
import os
from PIL import Image
import cv2

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]
def main():
    sg.theme('LightGreen5')
    layout = [
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Threshold Image"),
        ],
    ]
    window = sg.Window("Image Viewer", layout)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Threshold Image":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                # image = Image.open(values["-FILE-"])
                result = threshold_red(filename)
                # image.thumbnail((400, 400))
                # bio = io.BytesIO()
                # result.save(bio, format="PNG")
                savename = './output_images/output_1.jpg'
                # savepath = os.path.join(os.getcwd(), savename)
                # print(savepath)

                cv2.imwrite(savename, result)
                sg.popup('Thresholded Image Saved')
##### NOT PREVIEWING THRESHOLDED IMAGE - ISSUE WITH LOAD FILEPATH. SAVES FINE ######
                # window["-IMAGE-"].update(filename=savepath)
    window.close()
if __name__ == "__main__":
    main()