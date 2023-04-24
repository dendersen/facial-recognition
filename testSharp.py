import cv2 as cv
from SRC.image.imageCapture import Camera
from SRC.image.imageEditor import linearSharpen, sharpen
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
Cam = Camera(0)

class SliderWindow(QWidget):
  update_image_signal = pyqtSignal()

  def __init__(self):
    super().__init__()
    layout = QVBoxLayout()

    self.strength_label = QLabel()
    self.strength_slider = QSlider(Qt.Horizontal)
    self.strength_slider.setMinimum(0)
    self.strength_slider.setMaximum(1500)
    self.strength_slider.setValue(150)
    self.strength_slider.valueChanged.connect(self.update_strength_label)
    self.update_strength_label()
    layout.addWidget(QLabel("Strength"))
    layout.addWidget(self.strength_slider)
    layout.addWidget(self.strength_label)

    self.threshold_label = QLabel()
    self.threshold_slider = QSlider(Qt.Horizontal)
    self.threshold_slider.setMinimum(0)
    self.threshold_slider.setMaximum(255)
    self.threshold_slider.setValue(200)
    self.threshold_slider.valueChanged.connect(self.update_threshold_label)
    self.update_threshold_label()
    layout.addWidget(QLabel("Threshold"))
    layout.addWidget(self.threshold_slider)
    layout.addWidget(self.threshold_label)

    self.amplification_label = QLabel()
    self.amplification_slider = QSlider(Qt.Horizontal)
    self.amplification_slider.setMinimum(0)
    self.amplification_slider.setMaximum(255)
    self.amplification_slider.setValue(160)
    self.amplification_slider.valueChanged.connect(self.update_amplification_label)
    self.update_amplification_label()
    layout.addWidget(QLabel("Amplification"))
    layout.addWidget(self.amplification_slider)
    layout.addWidget(self.amplification_label)

    self.update_image_signal.connect(self.update_image)

    self.setLayout(layout)

  def update_strength_label(self):
    self.strength_label.setText(f"Strength: {self.strength_slider.value() / 100:.2f}")

  def update_threshold_label(self):
    self.threshold_label.setText(f"Threshold: {self.threshold_slider.value()}")

  def update_amplification_label(self):
    self.amplification_label.setText(f"Amplification: {self.amplification_slider.value() / 100:.2f}")

  def update_image(self):
    global pic
    strength = self.strength_slider.value() / 100
    threshold = self.threshold_slider.value()
    amplification = self.amplification_slider.value() / 100
    sharpen(pic, strength=strength, threshold=threshold, showSteps=True, showEnd=True, amplification=amplification)

app = QApplication([])
slider_window = SliderWindow()
slider_window.show()

class CameraThread(QThread):
  def run(self):
    global pic
    while True:
      # while cv.waitKey(10) != 32:
      pic = Cam.readCam(False)
      # cv.imshow("Cam output: ", pic)

      # if type(pic) != type(None):
      #   pic = Cam.processFace(pic, info=False)
      if type(pic) != type(None):
        cv.imshow("Cam output: ", pic)
        slider_window.update_image_signal.emit()
        # else:
        #   continue
      else:
        continue
      if cv.waitKey(20) == 27:
        exit()

camera_thread = CameraThread()
camera_thread.start()

app.exec_()
