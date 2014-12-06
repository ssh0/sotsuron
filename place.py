#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto, December 2014.

from Tkinter import *
import meeting

class Window(object):

    def __init__(self):
        self.root = Tk()
        canvas = Canvas(self.root, width=640, height=480)
        self.var = StringVar()
        self.oval(canvas)
        canvas.pack()
        label = Label(self.root, textvariable=self.var, font='Ubuntu 9')
        label.pack()

    def oval(self, canvas):
        for x in range(80, 560, 60):
            Oval(canvas, x, self.var)

    def display(self):
        self.root.mainloop()


class Oval:

    def __init__(self, canvas, x, var):
        self.c = canvas
        self.var = var
        self.id = self.c.create_oval(x, 220, x+40, 260, outline='', fill='#069')
        self.c.tag_bind(self.id, '<Button-1>', self.pressed)
        self.c.tag_bind(self.id, '<Button1-Motion>', self.dragging)
        self.c.bind('<Motion>', self.pointer)

    def pressed(self, event):
        self.x = event.x
        self.y = event.y

    def dragging(self, event):
        self.c.move(self.id, event.x - self.x, event.y - self.y)
        self.x = event.x
        self.y = event.y

    def pointer(self, event):
        self.var.set("       (%d,%d)" % (event.x, event.y))

if __name__ == '__main__':

    app = Window()
    app.display()


