# -*- coding = utf-8 -*-
# @Time : 2022/6/2 14:30
# @Author : fan
# @File:visual.py
# @Software: PyCharm

from tkinter import *
import tkinter.filedialog
import cv2
from stitch import stitch
from PIL import Image, ImageTk
import numpy as np

image_path = {}
stitched_image = 0


def run1():
    """
    输出选择的blending模式
    """
    s = var.get()
    txt.insert(END, 'you choose mode '+s+'\n')


def save():
    """
    保存最终拼接的全景图
    """
    # 打开文件对话框，选取路径
    filename = tkinter.filedialog.asksaveasfilename(filetypes=[('PNG','.png')])
    if filename != '':
        txt.insert(END, 'Image has been saved to ' + filename+'\n')
    else:
        txt.insert(END, 'You choose no image!'+'\n')
    # 先由cv2的image转换成PIL Image，再进行保存
    Image.fromarray(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGBA)).save(filename)


def help():
    """
    Help菜单栏的信息
    """
    # 打开新的窗口
    winNew = Toplevel(root)
    winNew.geometry('320x320')
    winNew.title('Help')
    
    # 输出提示信息到文本框
    txt2 = Text(winNew)
    txt2.place(relx=0.05, rely=0.2, relwidth=0.90, relheight=0.4)
    txt2.insert(END, 'You can try this:\nPlease read our report.\nOr contact us via student email.')
    
    # 关闭按钮
    btClose = Button(winNew, text='Close', command=winNew.destroy)
    btClose.place(relx=0.4, relwidth=0.2, rely=0.7, relheight=0.2)


def popupmenu(event):
    """
    鼠标右键弹出菜单的函数
    """
    mainmenu.post(event.x_root, event.y_root)


def choosefile(imageid):
    """
    选取原始图片
    """
    global image_path
    # 打开文件对话框，选取一张原始图片
    filename = tkinter.filedialog.askopenfilename()
    if filename != '':
        txt.insert(END, 'Your choose image '+filename+'\n')
    else:
        txt.insert(END, 'You choose no image!\n')
    # 保存文件目录到全局字典image_path
    image_path[imageid] = filename


class ImageDisplayer:
    """
    展示程序结果的类，可展示三张图片：原始图片并排放置、原始图片并排放置并标记特征及对应特征连线、最终拼接成品图
    """
    def __init__(self, master, images):
        # 打开新窗口并设置大小和标题
        self.window = Toplevel(master)
        self.window.geometry('960x960')
        self.window.title('Result Show')
        # 创建一个frame
        Frame(self.window, borderwidth=0, highlightthickness=0, height=20, width=30, bg='white').pack()
        # 在frame上放置两个label，用于显示图片标题和图片
        self.caption_label = Label(self.window, highlightthickness=0, borderwidth=0)
        self.caption_label.pack()
        self.image_label = Label(self.window, highlightthickness=0, borderwidth=0)
        self.image_label.pack()
        # 设置三个按钮：next、prev、close，分别用于下一张、上一张和关闭窗口
        self.next = Button(self.window, command=self.next_image, text="Next image", width=17, default=ACTIVE,
                           borderwidth=0)
        self.next.pack()
        self.prev = Button(self.window, command=self.prev_image, text="Prev image", width=17, default=ACTIVE,
                           borderwidth=0)
        self.prev.pack()
        self.close = Button(self.window, text='Close', command=self.window.destroy)
        self.close.pack()
        # 记录当前位于第几张
        self.cur_id = 0
        # 保存所有图片：共三张：原始图片并排放置、原始图片并排放置并标记特征及对应特征连线、最终拼接成品图
        self.images = images
        # 三张图片对应的标题
        self.captions = ["The original images", "The matched points of two images", "The stitched image"]

    def display_image(self):
        """
        展示图片
        """
        # 先将cv2数组转化为PIL Image，再转化为tkinter支持的图片类型
        self.imgt = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.images[self.cur_id], cv2.COLOR_BGR2RGBA)))
        # 更新图片和标题label，完成显示
        self.image_label.configure(image=self.imgt)
        self.caption_label.configure(text=self.captions[self.cur_id])

    def next_image(self):
        """
        执行下一张按钮的指令
        """
        # 更改self.cur_id变量后调用展示图片的函数
        if self.cur_id == len(self.images)-1:
            return
        self.cur_id += 1
        self.window.after(10, self.display_image)

    def prev_image(self):
        """
        执行上一张按钮的指令
        """
        if self.cur_id == 0:
            return
        # 更改self.cur_id变量后调用展示图片的函数
        self.cur_id -= 1
        self.window.after(10, self.display_image)

    def show_first_image(self):
        """
        展示第一张图片（初始化时用）
        """
        self.cur_id = 0
        self.window.after(10, self.display_image)


def make():
    """
    拼接操作主函数
    """
    global stitched_image
    image_name_1 = image_path[1]
    image_name_2 = image_path[2]
    
    # 读取两张图片文件并转化为cv2类型（因cv2不能读取带中文的路径，因此先用PIL的Image读取）
    image_left = cv2.cvtColor(np.array(Image.open(image_name_1)), cv2.COLOR_RGB2BGR)
    image_right = cv2.cvtColor(np.array(Image.open(image_name_2)), cv2.COLOR_RGB2BGR)

    # 调用stitch函数完成拼接
    blending_mode = var.get()  # 获取用户指定的blending模式：noBlending/linearBlending
    # stitch函数返回三张图片，分别是：原始图片并排放置、原始图片并排放置并标记特征及对应特征连线、最终拼接成品图
    matching_image_origin, matching_image, stitched_image = stitch(image_left, image_right, blending_mode, txt=txt)
    
    # 使用上面定义的类展示结果
    image_displayer = ImageDisplayer(root, [matching_image_origin, matching_image, stitched_image])
    image_displayer.show_first_image()


if __name__ == "__main__":
    # 创建tkinter主窗口
    root = Tk()
    root.title('Image tailor')
    root.geometry('640x640')
    
    # 在主窗口上创建标签以描述该程序功能
    lb = Label(root, text='A tailor for two images',\
               bg='#99cccc', fg='#ffcc99',\
               font=('Times', 28, 'bold'),\
               relief=FLAT)
    lb.place(relx=0.1, y=10, height=60, relwidth=0.8)
    
    # 提示用户选择blending模式
    lb2 = Label(root, text='Please choose one blending mode',\
               bg='#99cccc', fg='#ffcccc',\
               font=('Times', 14, 'bold'),\
               relief=FLAT)
    lb2.place(relx=0.2, rely=0.15, relheight=0.1, relwidth=0.6)
    
    # 使用单选按钮来获取用户选定的blending模式
    var = StringVar()
    rd1 = Radiobutton(root, text='No Blending', variable=var, value='no_blending', command=run1)
    rd1.place(relx=0.15, rely=0.25)
    rd2 = Radiobutton(root, text='Linear Blending', variable=var, value='linear_blending', command=run1)
    rd2.place(relx=0.65, rely=0.25)
    var.set("No Blending")
    
    # 选择图片的按钮，调用上面的选择图片函数
    btn1 = Button(root, text='Choose First Image', command=lambda: choosefile(1))
    btn1.place(relx=0.1, relwidth=0.3, rely=0.32, relheight=0.1)
    btn2 = Button(root, text='Choose Second Image', command=lambda: choosefile(2))
    btn2.place(relx=0.6, relwidth=0.3, rely=0.32, relheight=0.1)
    
    # 在主窗口增加文本框，以输出程序运行信息
    txt = Text(root)
    txt.place(rely=0.6, relheight=0.4)
    
    # 为主窗口增加菜单项，方面用户使用
    mainmenu = Menu(root)
    menuFile = Menu(mainmenu, tearoff=False)
    
    # 文件菜单项下设保存、退出两个功能
    mainmenu.add_cascade(label="File", menu=menuFile)
    menuFile.add_command(label='Save', command=save)
    menuFile.add_separator()
    menuFile.add_command(label='Exit', command=root.destroy)
    
    # 帮助菜单项下设帮助功能
    menuHelp = Menu(mainmenu, tearoff=False)
    mainmenu.add_cascade(label='Help', menu=menuHelp)
    menuHelp.add_command(label='Help', command=help)
    root.config(menu=mainmenu)
    
    # 绑定到右键菜单
    root.bind('<Button-3>', popupmenu)
    
    # 主要功能：图片拼接的入口按钮
    btn3 = Button(root, text='START STITCHING!', command=make)
    btn3.place(relx=0.375, relwidth=0.3, rely=0.45, relheight=0.1)
    
    root.mainloop()
