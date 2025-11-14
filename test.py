class SampShowWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(SampShowWindow, self).__init__(parent)
        self.setupUi(self)

        # 设置GroupBox固定大小
        self.groupimages.setFixedSize(650, 650)

        # 创建网格布局
        self.grid_layout = QGridLayout()
        self.grid_layout.setHorizontalSpacing(50)
        self.grid_layout.setVerticalSpacing(50)
        self.grid_layout.setContentsMargins(50, 50, 50, 50)

        # 初始化图片标签列表
        self.image_labels = []
        for _ in range(9):
            label = QLabel()
            self.image_labels.append(label)
            self.grid_layout.addWidget(label, _ // 3, _ % 3)

        # 设置布局到GroupBox
        self.groupimages.setLayout(self.grid_layout)

        # 添加"未选择类别"提示
        self.placeholder_label = QLabel("未选择类别", self.groupimages)
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 24px; color: gray;")
        self.placeholder_label.setGeometry(0, 0, self.groupimages.width(), self.groupimages.height())
        self.placeholder_label.show()

        # 获取images文件夹下的所有子文件夹
        self.image_folder = "Sample Set"

        # 创建映射字典：按钮英文名 -> (中文文件夹名, 颜色数组)
        self.mapping = {
            "curvedune": ("曲线型沙丘", [31, 119, 180]),
            "slopestreak": ("斜坡条纹", [44, 160, 44]),
            "texture": ("纹理", [127, 127, 127]),
            "rough": ("粗糙", [140, 86, 74]),
            "mixture": ("混合", [148, 103, 189]),
            "channel": ("沟槽", [152, 223, 138]),
            "straightdune": ("直线型沙丘", [174, 199, 232]),
            "mound": ("土丘", [196, 156, 148]),
            "ridge": ("山脊", [197, 176, 213]),
            "gully": ("冲沟", [214, 39, 40]),
            "craters": ("陨石坑群", [227, 119, 194]),
            "smooth": ("光滑", [247, 182, 210]),
            "cliff": ("悬崖", [255, 127, 14]),
            "mass": ("滑坡", [255, 152, 150]),
            "crater": ("陨石坑", [255, 187, 120])
        }

        # 创建按钮列表
        self.category_buttons = [
            self.btncurvedune, self.btnsloopestreak, self.btntexture,
            self.btnrough, self.btnmixture, self.btnchannel,
            self.btnstraightdune, self.btnmound, self.btnridge,
            self.btngully, self.btncraters, self.btnsmooth,
            self.btncliff, self.btnmass, self.btncrater
        ]

        # 添加label列表
        self.color_labels = []

        # 按钮绑定对应名称文件夹
        for button in self.category_buttons:
            key = button.objectName()[3:]

            if key in self.mapping:
                chinese_name, color_array = self.mapping[key]
                folder_path = os.path.join(self.image_folder, chinese_name)

                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    # 创建对应的label
                    color_label = QLabel(button.parent())
                    color_label.setObjectName(f"lbl{key}")
                    color_label.setFixedSize(61, 21)

                    # 设置颜色（将RGB数组转换为十六进制颜色代码）
                    hex_color = f"#{color_array[0]:02x}{color_array[1]:02x}{color_array[2]:02x}"
                    color_label.setStyleSheet(f"background-color: {hex_color}; border: 1px solid black;")

                    # 将label放置在按钮右侧间隔40像素的位置
                    button_x = button.x()
                    button_width = button.width()
                    color_label.move(button_x + button_width + 40, button.y())
                    color_label.show()

                    self.color_labels.append(color_label)

                    button.clicked.connect(
                        lambda checked, c=chinese_name, col=color_array: self.load_category(c, col)
                    )
                    button.setText(chinese_name)
                    button.show()
                else:
                    button.hide()
            else:
                button.hide()

        # 设置lblpages文本居中显示
        self.lblpages.setAlignment(Qt.AlignCenter)  # 水平和垂直都居中
        self.lblpages.setStyleSheet("font-size: 24px;")

        # 设置lbltotal文本显示
        self.lbltotal.setStyleSheet("font-size: 24px;")

        # 初始化图片变量
        self.current_category = None
        self.image_files = []
        self.total_pages = 0
        self.current_page = 0

        # 连接翻页按钮信号
        self.imgup.clicked.connect(self.prev_page)
        self.imgdown.clicked.connect(self.next_page)
        self.btnclear.clicked.connect(self.clear_display)

    def get_categories(self):
        """获取images文件夹下的所有子文件夹"""
        categories = []
        if os.path.exists(self.image_folder):
            for item in os.listdir(self.image_folder):
                item_path = os.path.join(self.image_folder, item)
                if os.path.isdir(item_path):
                    categories.append(item)
        return sorted(categories)

    def load_category(self, chinese_name, color_array):
        """加载指定类别的图片"""
        self.current_category = chinese_name
        self.current_color = color_array
        folder_path = os.path.join(self.image_folder, chinese_name)

        # 获取该类别下的所有图片
        self.image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")) +
                                  glob.glob(os.path.join(folder_path, "*.jpg")))
        self.total_pages = (len(self.image_files) + 8) // 9
        self.current_page = 0

        # 更新样本总量显示
        self.lbltotal.setText(f"样本总量:{len(self.image_files)}")

        # 隐藏"未选择类别"提示
        self.placeholder_label.hide()

        # 加载第一页
        self.load_page()

    def load_page(self):
        """加载当前页的图片"""
        # 更新页码显示
        self.lblpages.setText(f"{self.current_page + 1}/{self.total_pages}")

        if not self.image_files:
            # 如果没有图片，显示提示
            for i in range(9):
                if i == 4:
                    self.image_labels[i].setText("该类别无图片")
                    self.image_labels[i].setStyleSheet("font-size: 18px; color: gray;")
                    self.image_labels[i].setAlignment(Qt.AlignCenter)
                else:
                    self.image_labels[i].clear()
            return

        start_index = self.current_page * 9
        end_index = min(start_index + 9, len(self.image_files))

        # 加载当前页的图片
        for i in range(9):
            if i < (end_index - start_index):
                pixmap = QPixmap(self.image_files[start_index + i])
                pixmap = pixmap.scaled(150, 150)
                self.image_labels[i].setPixmap(pixmap)
            else:
                self.image_labels[i].clear()

    def prev_page(self):
        """上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.load_page()

    def next_page(self):
        """下一页"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.load_page()

    def clear_display(self):
        """清空图片显示和统计信息"""
        # 重置页码显示
        self.lblpages.setText("0/0")

        # 重置样本总量显示
        self.lbltotal.setText("样本总量:0")

        # 清空图片显示
        for label in self.image_labels:
            label.clear()

        # 显示"未选择类别"提示
        self.placeholder_label.show()

        # 重置内部状态变量
        self.current_category = None
        self.image_files = []
        self.total_pages = 0
        self.current_page = 0