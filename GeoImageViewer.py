# graphics_view.py（只展示关键片段）
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QRubberBand, QFileDialog, QMessageBox, QDialog, QLabel,
    QSpinBox, QGraphicsRectItem,QGraphicsItem, QWidget,
    QHBoxLayout, QPushButton, QFormLayout, QVBoxLayout
)
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import  QPixmap, QImage, QPainter, QColor
import os
import numpy as np


class ImageView(QGraphicsView):
    # 对外发射裁切结果（QImage）
    cropped = pyqtSignal(QImage, QImage, QImage, str)  # 原始图像, 掩膜, 叠加图像, 基础文件名
    # 添加鼠标移动信号
    mouseMoveSignal = pyqtSignal(float, float)  # 鼠标在图像上的坐标

    def __init__(self, parent=None, auto_save=True):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixItem = None
        self._orig_pixmap = None

        # rubberband
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._rubber_origin = QPoint()
        self._rubber_showing = False

        # 行为：左键平移，滚轮缩放
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.auto_save = bool(auto_save)
        self._overlay_data = None

        # interactive rectangle editing
        self._edit_rect_item = None
        self._edit_panel = None  # floating confirm/cancel panel widget

    def set_overlay_data(self, overlay_data):
        """设置掩膜数据"""
        self._overlay_data = overlay_data

    def loadPixmap(self, pixmap: QPixmap):
        """把一个 QPixmap 放到场景并缓存原始图像（原始分辨率用于裁切）。"""
        self.scene().clear()
        self.pixItem = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixItem)
        self._orig_pixmap = pixmap
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        # 清除存在的任何编辑框
        self._clear_edit_rect()

    def emit_mouse_position(self, event):
        """发射鼠标在图像上的坐标"""
        # 获取鼠标在视图中的位置
        pos = event.pos()
        # 如果有场景和图像项，计算在图像上的位置
        if self.scene() and self.pixItem:
            # 将视图坐标转换为场景坐标
            scene_pos = self.mapToScene(pos)
            # 将场景坐标转换为图像项坐标
            image_pos = self.pixItem.mapFromScene(scene_pos)
            # 发射信号，包含图像的x和y坐标
            self.mouseMoveSignal.emit(image_pos.x(), image_pos.y())
        else:
            # 如果没有图像项，发射视图坐标
            self.mouseMoveSignal.emit(pos.x(), pos.y())

    def mouseMoveEvent(self, event):
        # 首先处理 rubber band 的移动
        if self._rubber_showing:
            rect = QRect(self._rubber_origin, event.pos()).normalized()
            self._rubber.setGeometry(rect)
            event.accept()
            # 即使有 rubber band，也发射鼠标位置信号
            self.emit_mouse_position(event)
            return

        # 调用父类方法处理默认的鼠标移动事件
        super().mouseMoveEvent(event)
        # 发射鼠标位置信号
        self.emit_mouse_position(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._rubber_origin = event.pos()
            self._rubber.setGeometry(QRect(self._rubber_origin, QSize()))
            self._rubber.show()
            self._rubber_showing = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._rubber_showing:
            # 保持原有的坐标计算逻辑不变
            vb_rect = self._rubber.geometry()
            self._rubber.hide()
            self._rubber_showing = False

            if vb_rect.width() < 4 or vb_rect.height() < 4:
                super().mouseReleaseEvent(event)
                return

            # viewport -> scene -> pixmap-local coords
            top_left_scene = self.mapToScene(vb_rect.topLeft())
            bottom_right_scene = self.mapToScene(vb_rect.bottomRight())
            scene_rect = QRectF(top_left_scene, bottom_right_scene).normalized()

            # Clip to pixmap scene rect
            pix_scene_rect = self.pixItem.sceneBoundingRect()
            scene_rect = scene_rect.intersected(pix_scene_rect)
            if scene_rect.isEmpty():
                super().mouseReleaseEvent(event)
                return

            # create editable resizable rect in the scene
            self._create_edit_rect(scene_rect)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _create_edit_rect(self, scene_rect: QRectF):
        """在 scene 中创建 ResizableRectItem 并显示编辑面板"""
        # clear any existing edit rect
        self._clear_edit_rect()

        bounds = self.pixItem.sceneBoundingRect()
        ritem = ResizableRectItem(self.scene().sceneRect(), scene_rect, bounds)

        # place rect: the item currently uses rect in local coords; we prefer top-level item (no parent)
        ritem.set_scene_rect(scene_rect)

        self.scene().addItem(ritem)
        ritem.setZValue(1000)
        self._edit_rect_item = ritem

        # create floating panel (Confirm / Cancel / Input) as child widget of the view (so it floats)
        panel = QWidget(self.viewport())
        layout = QHBoxLayout(panel)
        btn_ok = QPushButton("确定")
        btn_cancel = QPushButton("取消")
        btn_input = QPushButton("输入数值")
        layout.addWidget(QLabel("裁剪:"))
        layout.addWidget(btn_input)
        layout.addStretch(1)
        layout.addWidget(btn_ok)
        layout.addWidget(btn_cancel)
        panel.setStyleSheet("background: rgba(30,30,30,200); color: white; padding:4px; border-radius:6px;")
        panel.adjustSize()
        # place at top-right corner of view's viewport
        margin = 8
        panel.move(self.viewport().width() - panel.width() - margin, margin)
        panel.show()

        btn_ok.clicked.connect(self._confirm_edit_rect)
        btn_cancel.clicked.connect(self._cancel_edit_rect)
        btn_input.clicked.connect(self._input_rect_dialog)

        self._edit_panel = panel

    def _clear_edit_rect(self):
        if getattr(self, "_edit_panel", None) is not None:
            try:
                self._edit_panel.hide();
                self._edit_panel.setParent(None)
            except Exception:
                pass
            self._edit_panel = None
        if getattr(self, "_edit_rect_item", None) is not None:
            try:
                self.scene().removeItem(self._edit_rect_item)
            except Exception:
                pass
            self._edit_rect_item = None

    def _input_rect_dialog(self):
        """弹出数值输入对话框以精确设置 x,y,w,h（像素坐标）"""
        # convert current scene rect to pixmap-local pixel coords
        if self._edit_rect_item is None: return
        scene_rect = self._edit_rect_item.get_scene_rect()
        # map scene rect to pixmap-local coordinates
        tl = self.pixItem.mapFromScene(scene_rect.topLeft())
        br = self.pixItem.mapFromScene(scene_rect.bottomRight())
        x = max(0, int(round(tl.x())));
        y = max(0, int(round(tl.y())))
        w = max(1, int(round(br.x() - tl.x())));
        h = max(1, int(round(br.y() - tl.y())))
        dlg = RectInputDialog(self, (x, y, w, h), max_w=self._orig_pixmap.width(), max_h=self._orig_pixmap.height())
        if dlg.exec() == QDialog.Accepted:
            fx, fy, fw, fh = dlg.rect()
            # clamp
            fw = max(1, min(fw, self._orig_pixmap.width() - fx))
            fh = max(1, min(fh, self._orig_pixmap.height() - fy))
            # map pixmap-local to scene coordinates for setting rect
            tl_scene = self.pixItem.mapToScene(QPointF(fx, fy))
            br_scene = self.pixItem.mapToScene(QPointF(fx + fw, fy + fh))
            new_scene_rect = QRectF(tl_scene, br_scene).normalized()
            # set into edit rect
            self._edit_rect_item.set_scene_rect(new_scene_rect)

    def _confirm_edit_rect(self):
        """从编辑矩形生成裁剪 QImage/掩膜/overlay，并发信号"""
        if self._edit_rect_item is None:
            return
        scene_rect = self._edit_rect_item.get_scene_rect()
        # Convert scene rect to pixmap-local coords (floating)
        tl_local = self.pixItem.mapFromScene(scene_rect.topLeft())
        br_local = self.pixItem.mapFromScene(scene_rect.bottomRight())
        x = max(0, int(round(tl_local.x())));
        y = max(0, int(round(tl_local.y())))
        w = int(round(br_local.x() - tl_local.x()))
        h = int(round(br_local.y() - tl_local.y()))
        if w <= 0 or h <= 0:
            QMessageBox.warning(self, "裁剪失败", "裁剪区域大小无效")
            return
        # Crop original QImage
        orig_qimage = self._orig_pixmap.toImage()
        cropped_qimage = orig_qimage.copy(x, y, w, h)

        # 生成掩膜和 overlay（如果有 self._overlay_data）
        cropped_mask_qimage = None
        cropped_overlay_qimage = None
        if getattr(self, "_overlay_data", None) is not None:
            try:
                ma = self._overlay_data
                mh, mw = ma.shape[:2]
                # ensure indices within mask bounds
                mx = min(x, mw - 1);
                my = min(y, mh - 1)
                mw_mask = min(w, mw - mx);
                mh_mask = min(h, mh - my)
                if mw_mask > 0 and mh_mask > 0:
                    sub = ma[my:my + mh_mask, mx:mx + mw_mask]
                    if sub.ndim == 3:
                        if sub.dtype != np.uint8: sub = sub.astype(np.uint8)
                        bytes_per_line = 3 * sub.shape[1]
                        cropped_mask_qimage = QImage(sub.tobytes(), sub.shape[1], sub.shape[0], bytes_per_line,
                                                     QImage.Format_RGB888)
                        # overlay
                        cropped_overlay_qimage = cropped_qimage.copy()
                        painter = QPainter(cropped_overlay_qimage)
                        painter.setOpacity(0.5)
                        if cropped_mask_qimage.width() != w or cropped_mask_qimage.height() != h:
                            painter.drawImage(0, 0, cropped_mask_qimage.scaled(w, h, Qt.IgnoreAspectRatio,
                                                                               Qt.SmoothTransformation))
                        else:
                            painter.drawImage(0, 0, cropped_mask_qimage)
                        painter.end()
                    else:
                        # single channel -> treat as grayscale mask
                        if sub.dtype != np.uint8: sub = sub.astype(np.uint8)
                        bytes_per_line = sub.shape[1]
                        cropped_mask_qimage = QImage(sub.tobytes(), sub.shape[1], sub.shape[0], bytes_per_line,
                                                     QImage.Format_Grayscale8)
                        cropped_overlay_qimage = cropped_qimage.copy()
                        painter = QPainter(cropped_overlay_qimage)
                        painter.setOpacity(0.5)
                        # simple white overlay for grayscale mask (or you may color it)
                        rgb_mask = QImage(mw_mask, mh_mask, QImage.Format_RGB888)
                        rgb_mask.fill(Qt.white)
                        if rgb_mask.width() != w or rgb_mask.height() != h:
                            painter.drawImage(0, 0,
                                              rgb_mask.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
                        else:
                            painter.drawImage(0, 0, rgb_mask)
                        painter.end()
            except Exception as e:
                print("生成掩膜/叠加出错:", e)

        # 防止 None 传递给强类型信号，回退为空 QImage
        mask_to_emit = cropped_mask_qimage if isinstance(cropped_mask_qimage, QImage) else QImage()
        overlay_to_emit = cropped_overlay_qimage if isinstance(cropped_overlay_qimage, QImage) else QImage()
        base_name = f"crop_{x}_{y}_{w}_{h}"

        if not self.auto_save:
            try:
                self.cropped.emit(cropped_qimage, mask_to_emit, overlay_to_emit, base_name)
            except Exception as e:
                print("发射信号时出错:", e)
        else:
            # 自动保存
            self.save_cropped_images(cropped_qimage,
                                     (cropped_mask_qimage if cropped_mask_qimage is not None else None),
                                     (cropped_overlay_qimage if cropped_overlay_qimage is not None else None),
                                     base_name)

            # 清理编辑 UI
        self._clear_edit_rect()

    def _cancel_edit_rect(self):
        self._clear_edit_rect()

    def save_cropped_images(self, orig_image, mask_image, overlay_image, base_name):
        """
        保存裁剪的图像（不处理 numpy）：只接受 QImage/QPixmap/对象带 save 方法 或 None。
        - 如果 mask_image is None: 仅保存 orig_image（不保存 mask/overlay，也不报错）。
        - 如果 mask_image 非 None: 尝试保存 mask 和 overlay（overlay None 时尝试基于 orig+mask 生成）。
        """
        parent_window = self.window() or self
        save_dir = QFileDialog.getExistingDirectory(parent_window, "选择保存文件夹", "")
        if not save_dir:
            return

        safe_base = str(base_name).replace(" ", "_")
        saved_files = []
        failed_files = []

        # ---------- 保存原始图像（必须保存） ----------
        orig_path = os.path.join(save_dir, f"{safe_base}_original.png")
        try:
            saved = False
            if isinstance(orig_image, QImage):
                saved = orig_image.save(orig_path)
            elif isinstance(orig_image, QPixmap):
                saved = orig_image.save(orig_path)
            else:
                # 兼容：对象若有 save 方法则试一试
                try:
                    saved = orig_image.save(orig_path)
                except Exception:
                    saved = False

            if saved:
                saved_files.append(orig_path)
            else:
                failed_files.append(orig_path)
        except Exception:
            failed_files.append(orig_path)

        # 如果原图保存失败——认为整体失败并返回
        if failed_files:
            QMessageBox.critical(parent_window, "保存失败",
                                 "无法保存原始裁剪图像：\n" + "\n".join(failed_files))
            return

        overlay_data = getattr(self, "_overlay_data", None)
        # 检查 overlay_data 是否为 None 或空数组
        if overlay_data is None or (hasattr(overlay_data, 'size') and overlay_data.size == 0):
            QMessageBox.information(parent_window, "保存成功",
                                    "已保存以下文件：\n" + "\n".join(saved_files))
            return

        mask_qimage = None
        if isinstance(mask_image, QImage):
            mask_qimage = mask_image
        elif isinstance(mask_image, QPixmap):
            mask_qimage = mask_image.toImage()
        else:
            # mask 存在但不是 QImage/QPixmap —— 记录失败但不要立刻弹窗中断
            mask_path = os.path.join(save_dir, f"{safe_base}_mask.png")
            failed_files.append(mask_path)
            mask_qimage = None  # 继续尝试 overlay（但没有 mask 无法生成 overlay）
        # 只有在确实拿到 mask_qimage 时才尝试写文件
        if isinstance(mask_qimage, QImage):
            mask_path = os.path.join(save_dir, f"{safe_base}_mask.png")
            try:
                if mask_qimage.save(mask_path):
                    saved_files.append(mask_path)
                else:
                    failed_files.append(mask_path)
            except Exception:
                failed_files.append(mask_path)

        # ---------- overlay：优先保存 overlay_image，否则尝试基于 orig+mask 生成 ----------
        overlay_qimage = None
        if isinstance(overlay_image, QImage):
            overlay_qimage = overlay_image
        elif isinstance(overlay_image, QPixmap):
            overlay_qimage = overlay_image.toImage()
        else:
            # overlay_image 为 None 或不支持类型 -> 尝试基于原图 + mask 生成（如果可能）
            try:
                orig_qimage = None
                if isinstance(orig_image, QImage):
                    orig_qimage = orig_image
                elif isinstance(orig_image, QPixmap):
                    orig_qimage = orig_image.toImage()
                if isinstance(orig_qimage, QImage) and isinstance(mask_qimage, QImage):
                    overlay_qimage = orig_qimage.copy()
                    painter = QPainter(overlay_qimage)
                    painter.setOpacity(0.5)
                    if mask_qimage.size() != overlay_qimage.size():
                        painter.drawImage(0, 0, mask_qimage.scaled(overlay_qimage.size(), Qt.IgnoreAspectRatio,
                                                                   Qt.SmoothTransformation))
                    else:
                        painter.drawImage(0, 0, mask_qimage)
                    painter.end()
            except Exception as e:
                # 生成 overlay 失败 -> 记录并继续
                print("生成 overlay 失败:", e)
                overlay_qimage = None

        # 仅在确实有 overlay_qimage 时才尝试保存
        if isinstance(overlay_qimage, QImage):
            overlay_path = os.path.join(save_dir, f"{safe_base}_overlay.png")
            try:
                if overlay_qimage.save(overlay_path):
                    saved_files.append(overlay_path)
                else:
                    failed_files.append(overlay_path)
            except Exception:
                failed_files.append(overlay_path)

        # ---------- 最终反馈：仅报告实际尝试但失败的文件 ----------
        if failed_files:
            QMessageBox.critical(parent_window, "部分保存失败",
                                 "下列文件保存失败：\n" + "\n".join(failed_files) +
                                 "\n\n已成功保存文件：\n" + "\n".join(saved_files))
        else:
            QMessageBox.information(parent_window, "保存成功",
                                    "已保存以下文件：\n" + "\n".join(saved_files))

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.2
        else:
            factor = 1 / 1.2
        self.scale(factor, factor)



class ResizeHandle(QGraphicsRectItem):
    """单个拖拽句柄（小方块）。父对象为 ResizableRectItem"""
    SIZE = 8.0

    def __init__(self, parent_rect, position_key: str):
        # position_key: 'tl','tr','bl','br','t','b','l','r'
        half = ResizeHandle.SIZE / 2.0
        super().__init__(-half, -half, ResizeHandle.SIZE, ResizeHandle.SIZE, parent=parent_rect)
        self.setBrush(QColor(255, 255, 255))
        self.setPen(QColor(80, 80, 80))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)  # 保持屏幕大小不随缩放变化
        self.position_key = position_key
        self.parent_rect = parent_rect
        self.setCursor(self._cursor_for_key(position_key))

    def _cursor_for_key(self, k):
        cursors = {
            'tl': Qt.SizeFDiagCursor, 'br': Qt.SizeFDiagCursor,
            'tr': Qt.SizeBDiagCursor, 'bl': Qt.SizeBDiagCursor,
            't': Qt.SizeVerCursor, 'b': Qt.SizeVerCursor,
            'l': Qt.SizeHorCursor, 'r': Qt.SizeHorCursor
        }
        return cursors.get(k, Qt.ArrowCursor)

    def itemChange(self, change, value):
        # 当句柄移动时，更新父矩形
        if change == QGraphicsItem.ItemPositionChange:
            parent = self.parent_rect
            # 如果父对象正在批量更新句柄位置，直接让默认处理，避免递归
            if getattr(parent, "_updating_handles", False):
                return super().itemChange(change, value)

            # value 是相对于父项的局部坐标（因为句柄是 parent 的 child）
            try:
                new_pos_local = value
                # 调用父对象处理句柄移动（父对象会根据 scene 坐标调整 rect）
                parent.handle_moved(self.position_key, new_pos_local)
            except Exception:
                # 出错时仍然返回默认行为，避免抛出异常中断交互
                pass
            # 返回当前实际位置（使用 self.pos() 保持句柄当前坐标，不受 value 引起的重复移动）
            return super().itemChange(change, self.pos())

        return super().itemChange(change, value)


class ResizableRectItem(QGraphicsRectItem):
    """
    可移动、可缩放的矩形（包含 8 个句柄）。
    rect 的坐标以 scene 坐标系管理（item pos/rect）。
    parent_scene_bounds: QRectF 表示允许矩形活动的边界（通常为 pixmap 的 sceneRect）
    """
    HANDLE_KEYS = ('tl','t','tr','r','br','b','bl','l')

    def __init__(self, scene_rect: QRectF, initial_rect: QRectF, parent_scene_bounds: QRectF):
        # We create the rect item in the scene coordinates by setting rect=initial_rect and pos=(0,0)
        super().__init__(initial_rect)
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setPen(QColor(0, 200, 255))
        self.setBrush(QColor(0, 0, 0, 0))
        self.handles = {}
        self.parent_scene_bounds = parent_scene_bounds
        # 锁，防止在我们批量设置 handles 的位置时触发 handle 的回调
        self._updating_handles = False
        for key in ResizableRectItem.HANDLE_KEYS:
            h = ResizeHandle(self, key)
            self.handles[key] = h
        # 初始放置
        self.update_handles_positions()

    def update_handles_positions(self):
        """把句柄放到当前 rect 的位置（使用 rect() 的 local 坐标）"""
        r = self.rect()
        coords = {
            'tl': QPointF(r.left(), r.top()),
            'tr': QPointF(r.right(), r.top()),
            'bl': QPointF(r.left(), r.bottom()),
            'br': QPointF(r.right(), r.bottom()),
            't': QPointF((r.left() + r.right()) / 2.0, r.top()),
            'b': QPointF((r.left() + r.right()) / 2.0, r.bottom()),
            'l': QPointF(r.left(), (r.top() + r.bottom()) / 2.0),
            'r': QPointF(r.right(), (r.top() + r.bottom()) / 2.0)
        }
        # 开始批量更新 -> 设锁
        self._updating_handles = True
        try:
            for k, h in self.handles.items():
                # setPos 会触发子项的 itemChange，但因为 _updating_handles=True，句柄会跳过回调
                h.setPos(coords[k])
        finally:
            # 解除锁
            self._updating_handles = False

    def handle_moved(self, key: str, new_pos_local: QPointF):
        """
        Called by a handle when it was moved (new_pos_local is in parent's local coordinates).
        We compute a new rect accordingly, constrain to parent_scene_bounds, and update rect and handles.
        """
        # Convert local coords to scene coords of the handle's center:
        handle_scene_pos = self.mapToScene(new_pos_local)
        # We'll compute new rect in scene coordinates, then setRect with rect in local coords.
        # Current rect in scene coords:
        current_scene_rect = self.mapToScene(self.rect()).boundingRect()  # QRectF in scene
        left = current_scene_rect.left()
        top = current_scene_rect.top()
        right = current_scene_rect.right()
        bottom = current_scene_rect.bottom()

        # depending on key, update corresponding edges
        if key == 'tl':
            left = handle_scene_pos.x(); top = handle_scene_pos.y()
        elif key == 'tr':
            right = handle_scene_pos.x(); top = handle_scene_pos.y()
        elif key == 'bl':
            left = handle_scene_pos.x(); bottom = handle_scene_pos.y()
        elif key == 'br':
            right = handle_scene_pos.x(); bottom = handle_scene_pos.y()
        elif key == 't':
            top = handle_scene_pos.y()
        elif key == 'b':
            bottom = handle_scene_pos.y()
        elif key == 'l':
            left = handle_scene_pos.x()
        elif key == 'r':
            right = handle_scene_pos.x()

        # Normalize (ensure left < right, top < bottom) and constrain to bounds
        new_scene_left = min(left, right)
        new_scene_right = max(left, right)
        new_scene_top = min(top, bottom)
        new_scene_bottom = max(top, bottom)

        # Constrain to parent_scene_bounds
        pb = self.parent_scene_bounds
        new_scene_left = max(pb.left(), new_scene_left)
        new_scene_right = min(pb.right(), new_scene_right)
        new_scene_top = max(pb.top(), new_scene_top)
        new_scene_bottom = min(pb.bottom(), new_scene_bottom)

        # Avoid zero-size: enforce minimum size
        MIN_SIZE = 2.0
        if new_scene_right - new_scene_left < MIN_SIZE:
            new_scene_right = new_scene_left + MIN_SIZE
        if new_scene_bottom - new_scene_top < MIN_SIZE:
            new_scene_bottom = new_scene_top + MIN_SIZE

        new_scene_rect = QRectF(QPointF(new_scene_left, new_scene_top), QPointF(new_scene_right, new_scene_bottom))

        # We must set our rect in local coordinates (rect relative to this item's transformation).
        # Simplest: set new rect to new_scene_rect but as rect with top-left at new_scene_rect.topLeft() mapped to parent local coordinates.
        parent_item = self.parentItem()
        if parent_item is None:
            # this item is top-level in scene; map scene->item coordinates:
            local_tl = self.mapFromScene(new_scene_rect.topLeft())
            local_br = self.mapFromScene(new_scene_rect.bottomRight())
            local_rect = QRectF(local_tl, local_br).normalized()
            self.setRect(local_rect)
            self.setPos(QPointF(0, 0))  # keep pos 0 because rect uses item-local coords
        else:
            # If this item has parent (unlikely here), handle differently
            local_tl = self.mapFromScene(new_scene_rect.topLeft())
            local_br = self.mapFromScene(new_scene_rect.bottomRight())
            local_rect = QRectF(local_tl, local_br).normalized()
            self.setRect(local_rect)

        # After setting rect, update handles positions
        self.update_handles_positions()

    def itemChange(self, change, value):
        # When the whole rect moves, constrain to parent_scene_bounds and update handles.
        if change == QGraphicsItem.ItemPositionChange:
            # new pos is in parent's coordinates; but we used rect to represent geometry, so moving may be limited.
            # To keep things simpler, instead of moving the item via pos, we will rely on setRect even for moving.
            pass
        if change == QGraphicsItem.ItemPositionHasChanged or change == QGraphicsItem.ItemTransformHasChanged:
            self.update_handles_positions()
        return super().itemChange(change, value)

    def set_scene_rect(self, new_scene_rect: QRectF):
        """直接以 scene 坐标设置矩形（通常用于从 rubberband 创建）"""
        # convert scene rect to local coordinates
        local_tl = self.mapFromScene(new_scene_rect.topLeft())
        local_br = self.mapFromScene(new_scene_rect.bottomRight())
        self.setRect(QRectF(local_tl, local_br).normalized())
        self.update_handles_positions()

    def get_scene_rect(self) -> QRectF:
        """返回当前矩形在 scene 坐标系中的 QRectF"""
        rect_local = self.rect()
        top_left_scene = self.mapToScene(rect_local.topLeft())
        bottom_right_scene = self.mapToScene(rect_local.bottomRight())
        return QRectF(top_left_scene, bottom_right_scene).normalized()

# ---------- 精确输入对话框（可选） ----------
class RectInputDialog(QDialog):
    def __init__(self, parent=None, initial_rect=(0,0,100,100), max_w=10000, max_h=10000):
        super().__init__(parent)
        self.setWindowTitle("输入裁剪矩形 (x,y,w,h)")
        x0,y0,w0,h0 = initial_rect
        form = QFormLayout()
        self.spin_x = QSpinBox(); self.spin_x.setMaximum(max_w); self.spin_x.setValue(x0)
        self.spin_y = QSpinBox(); self.spin_y.setMaximum(max_h); self.spin_y.setValue(y0)
        self.spin_w = QSpinBox(); self.spin_w.setMaximum(max_w); self.spin_w.setValue(w0)
        self.spin_h = QSpinBox(); self.spin_h.setMaximum(max_h); self.spin_h.setValue(h0)
        form.addRow("X:", self.spin_x)
        form.addRow("Y:", self.spin_y)
        form.addRow("宽 (w):", self.spin_w)
        form.addRow("高 (h):", self.spin_h)
        ok = QPushButton("确定"); cancel = QPushButton("取消")
        ok.clicked.connect(self.accept); cancel.clicked.connect(self.reject)
        btns = QHBoxLayout(); btns.addStretch(1); btns.addWidget(ok); btns.addWidget(cancel)
        lay = QVBoxLayout(); lay.addLayout(form); lay.addLayout(btns)
        self.setLayout(lay)
    def rect(self):
        return (self.spin_x.value(), self.spin_y.value(), self.spin_w.value(), self.spin_h.value())