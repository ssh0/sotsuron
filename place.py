#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto, December 2014.

from Tkinter import *
import numpy as np
import meeting


class Main:

    def __init__(self):
        N = 6
        app = meeting.meeting(N)
        app.radius = 1/3.
        window = Window(N, main=app)
        window.display()

class Window(object):

    def __init__(self, N, main):
        self.root = Tk()
        self.main = main
        self.width = 640
        self.height = 480
        self.canvas = Canvas(self.root, width=self.width, height=self.height)
        self.var = StringVar()
        self.oval(self.canvas, N)
        self.canvas.bind('<Motion>', self.pointer)
        self.canvas.pack()
        label = Label(self.root, textvariable=self.var, font='Ubuntu 9')
        label.pack(side='left')
        b1 = Button(self.root, text='start', command=self.b1_clicked)
        b1.pack(side='right')
        b2 = Button(self.root, text='save', command=self.b2_clicked)
        b2.pack(side='right')

    def oval(self, canvas, N=6):
        self.members = dict()
        deg = np.linspace(0., 360., N, endpoint=False)
        radius = 20
        self.r = int((min(self.height, self.width)/2-radius)*0.9)
        self.centerx = int(self.width/2)
        self.centery = int(self.height/2)
        for n in range(1, N+1):
            rad = np.radians(deg[n-1])
            self.members[n] = Oval(canvas, n,
                                   self.centerx+self.r*np.cos(rad),
                                   self.centery+self.r*np.sin(rad),
                                   radius, self.var)

    def pointer(self, event):
        self.var.set("(%d,%d)" % (event.x, event.y))

    def b1_clicked(self):
        silent = meeting.Person(place=(0., 0.))
        silent.ideas = [0.5]
        def distance_silent(p):
            d = 10.
            return d
        silent.distance = distance_silent
        self.main.members = {0: silent, }
        for n in range(1, self.main.N+1):
            x = (self.members[n].x-self.centerx)/float(self.r)
            y = (self.members[n].y-self.centery)/float(self.r)
            self.main.members[n] = meeting.Person(place=(x, y))
        self.main.progress()

    def b2_clicked(self):
        import tkFileDialog
        import os

        fTyp = [('eps file', '*.eps'), ('all files', '*')]
        filename = tkFileDialog.asksaveasfilename(filetypes=fTyp,
                                                  initialdir=os.getcwd(),
                                                  initialfile='figure_1.eps')

        if filename is None:
            return
        try:
            self.canvas.postscript(file=filename)
        except TclError:
            print """
            TclError: Cannot save the figure.
            Canvas Window must be alive for save."""
            return 1

    def display(self):
        self.root.mainloop()


class Oval:

    def __init__(self, canvas, id, x, y, r, var):
        self.c = canvas
        self.x = x
        self.y = y
        self.var = var
        self.tag = str(id)
        self.c.create_oval(x-r, y-r, x+r, y+r, outline='', fill='#069', tags=self.tag)

        self.c.tag_bind(self.tag, '<Button-1>', self.pressed)
        self.c.tag_bind(self.tag, '<Button1-Motion>', self.dragging)

    def pressed(self, event):
        self.x = event.x
        self.y = event.y

    def dragging(self, event):
        self.c.move(self.tag, event.x - self.x, event.y - self.y)
        self.x = event.x
        self.y = event.y

if __name__ == '__main__':

    main = Main()


