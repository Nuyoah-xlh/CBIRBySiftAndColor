# @Author   : 919106840638肖林航
# @time     : 2021/10/08 上午09:30
# @Software : PyCharm


from SelectAndSearch import *

#  创建主界面
app = tk.Tk()
app.title('基于内容的图像检索（CBIR）')
background = tk.PhotoImage(file="icon/background.gif")  # 背景图片

#  添加背景和标题
bg = tk.Label(app, image=background, compound=tk.CENTER,bg="#989cb8")
bg.place(relx=0, rely=0, relwidth=1, relheight=1)
title = tk.Label(app, text='基于内容的图像检索（CBIR）', font=("宋体", 25), image=background, compound=tk.CENTER)
title.place(relx=0.3, rely=0.2, relwidth=0.5, relheight=0.1)

title = tk.Label(app, text='说明：基于颜色特征检索为前期所用算法，因检索效率低、\n'
                           '检索效果一般，进而使用基于SIFT特征检索为主要研究方向。\n'
                           '这里基于颜色特征检索用了前10000张图像仅作为对照来详细研\n'
                           '究基于SIFT特征的图像检索。', font=("宋体", 12), background="#f8caa8", compound=tk.CENTER)
title.place(relx=0.25, rely=0.7, relwidth=0.4, relheight=0.2)

SelectAndSearch(app)
app.mainloop()
