# import tkinter as tk
# import PIL.Image
# import PIL.ImageTk
# from tkinter import *
#
# class ImageCanvas(Frame):
#     def __init__(self, master):
#         Frame.__init__(self, master)
#         self.grid_rowconfigure(0, weight=1)
#         self.grid_columnconfigure(0, weight=1)
#         self.canvas = Canvas(self, width=720, height=480, bd=0)
#         self.canvas.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
#
#         self.root = tk.Tk()
#         self._job = NONE
#         self.slider = tk.Scale(self.root, from_=0, to=3, orient = "horizontal", command=self.updateValue)
#         self.slider.pack()
#         # self.root.mainloop()
#
#     def updateValue(self, event):
#         if self._job:
#             self.root.after_cancel(self._job)
#         self._job = self.root.after(500, self.result)
#
#     def result(self):
#         self._job=None
#         print(self.slider.get())
#         returnedValue = self.slider.get()
#         return returnedValue
#
# class ImgTk(tk.Tk):
#     def __init__(self):
#         tk.Tk.__init__(self)
#         self.main = ImageCanvas(self)
#         self.main.grid(row=0, column=0, sticky='nsew')
#         self.c = self.main.canvas
#
#         self.currentImage = {}
#         self.load_imgfile(images[self.main.result()])
#
#     def load_imgfile(self, filename):
#         self.img = PIL.Image.open(filename)
#         self.currentImage['data'] = self.img
#
#         self.photo = PIL.ImageTk.PhotoImage(self.img)
#         self.c.xview_moveto(0)
#         self.c.yview_moveto(0)
#         self.c.create_image(0, 0, image=self.photo, anchor='nw', tags='img')
#         self.c.config(scrollregion=self.c.bbox('all'))
#         self.currentImage['photo'] = self.photo
#
# images = ['/home/imerse/Documents/spring22/CSDL/project/dVRK_segmentation/data/ucl_dataset/Video_01/images/000.png', '/home/imerse/Documents/spring22/CSDL/project/dVRK_segmentation/data/ucl_dataset/Video_01/images/001.png', '/home/imerse/Documents/spring22/CSDL/project/dVRK_segmentation/data/ucl_dataset/Video_01/images/002.png']
#
# root = tk.Tk()
# app = ImgTk()
# root.mainloop()

from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)

def plot():

    # the figure that will contain the plot
    fig = Figure(figsize = (5, 5),
                 dpi = 100)

    # list of squares
    y = [i**2 for i in range(101)]

    # adding the subplot
    plot1 = fig.add_subplot(111)

    # plotting the graph
    plot1.plot(y)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master = window)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()

# the main Tkinter window
window = Tk()

# setting the title
window.title('Plotting in Tkinter')

# dimensions of the main window
window.geometry("500x500")

# button that displays the plot
plot_button = Button(master = window,
                     command = plot,
                     height = 2,
                     width = 10,
                     text = "Plot")

# place the button
# in main window
plot_button.pack()

# run the gui
window.mainloop()
