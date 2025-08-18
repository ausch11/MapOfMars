# graphics_view.py
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt

class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        # 拖拽平移模式
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # 滚轮缩放以光标所在点为锚
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def loadPixmap(self, pixmap):
        """把一个 QPixmap 放到场景并适应窗口大小。"""
        self.scene().clear()
        self.pixItem = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixItem)
        # 初次加载时让它铺满视图
        self.fitInView(self.pixItem, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        """重写滚轮：放大/缩小视图。"""
        if event.angleDelta().y() > 0:
            factor = 1.2
        else:
            factor = 1 / 1.2
        self.scale(factor, factor)
