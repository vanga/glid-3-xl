from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QPoint, QRect, QBuffer
import PyQt5.QtGui as QtGui

class Draw(QMainWindow):
    def __init__(self, width, height, im):
        super().__init__()
        self.drawing = False
        self.lastPoint = QPoint()

        self.qim = QtGui.QImage(im.tobytes(
            "raw", "RGB"), im.width, im.height, QtGui.QImage.Format_RGB888)
        self.image = QtGui.QPixmap.fromImage(self.qim)

        canvas = QtGui.QImage(im.width, im.height,
                              QtGui.QImage.Format_ARGB32)
        self.canvas = QtGui.QPixmap.fromImage(canvas)
        self.canvas.fill(Qt.transparent)

        self.setGeometry(0, 0, im.width, im.height)
        self.resize(self.image.width(), self.image.height())
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(
            QRect(0, 0, self.image.width(), self.image.height()), self.image)
        painter.drawPixmap(
            QRect(0, 0, self.canvas.width(), self.canvas.height()), self.canvas)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.canvas)
            painter.setPen(QPen(Qt.red, (self.width()+self.height()) /
                                20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

    def getCanvas(self):
        image = self.canvas.toImage()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        image.save(buffer, "PNG")
        pil_im = Image.open(io.BytesIO(buffer.data()))
        return pil_im

    def resizeEvent(self, event):
        self.image = QtGui.QPixmap.fromImage(self.qim)
        self.image = self.image.scaled(self.width(), self.height())

        canvas = QtGui.QImage(
            self.width(), self.height(), QtGui.QImage.Format_ARGB32)
        self.canvas = QtGui.QPixmap.fromImage(canvas)
        self.canvas.fill(Qt.transparent)
