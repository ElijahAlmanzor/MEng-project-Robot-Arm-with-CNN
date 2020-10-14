#-------------------------------------------------------------------------------
# Name:        Object bounding box label tool (with angle)
# Purpose:     Label object bboxes for Grasp Detection
# Author:      YuHsuan Yen
# Created:     2016/12/30
# Demo Video: https://youtu.be/dZGoISfAJmI
#-------------------------------------------------------------------------------
from __future__ import division
from Tkinter import *
import tkMessageBox
from PIL import Image, ImageTk
import os
import glob
import random
import math as m
import cmath as cm
import numpy as np
# colors for the bboxes
COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']
# image sizes for the examples
SIZE = 480, 640

class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = FALSE, height = FALSE)

        # initialize global state
        self.imageDir = ''
        self.imageList= []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0
        self.STATE['gR'] = []
        self.STATE['gR_deg'] = 0
        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text = "Image Dir:")
        self.label.grid(row = 0, column = 0, sticky = E)
        self.entry = Entry(self.frame)
        self.entry.grid(row = 0, column = 1, sticky = W+E)
        self.ldBtn = Button(self.frame, text = "Load", command = self.loadDir)
        self.ldBtn.grid(row = 0, column = 2, sticky = W+E)

        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove) 
        self.parent.bind("<Escape>", self.cancelBBox)   # press <Espace> to cancel current bbox
        self.parent.bind("<Delete>", self.delBBox)      # press <Delete> to cancel the selection
        self.parent.bind("<Prior>", self.prevImage)        # press <up> to go backforward
        self.parent.bind("<Next>", self.nextImage)      # press <down> to go forward
        # self.parent.bind("<Home>",self.loadDir)        # press <Enter> to load dir
        self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

        # showing bbox info & delete bbox
        self.lb1 = Label(self.frame, text = 'Bounding boxes:')
        self.lb1.grid(row = 1, column = 2,  sticky = W+N)
        self.listbox = Listbox(self.frame, width = 28, height = 12)
        self.listbox.grid(row = 2, column = 2, sticky = N)
        self.btnDel = Button(self.frame, text = 'Delete', command = self.delBBox)
        self.btnDel.grid(row = 3, column = 2, sticky = W+E+N)
        self.btnClear = Button(self.frame, text = 'ClearAll', command = self.clearBBox)
        self.btnClear.grid(row = 4, column = 2, sticky = W+E+N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 5, column = 1, columnspan = 2, sticky = W+E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width = 10, command = self.prevImage)
        self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width = 10, command = self.nextImage)
        self.nextBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
        self.progLabel.pack(side = LEFT, padx = 5)
        self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
        self.tmpLabel.pack(side = LEFT, padx = 5)
        self.idxEntry = Entry(self.ctrPanel, width = 5)
        self.idxEntry.pack(side = LEFT)
        self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
        self.goBtn.pack(side = LEFT)

        # example pannel for illustration
        self.egPanel = Frame(self.frame, border = 10)
        self.egPanel.grid(row = 1, column = 0, rowspan = 5, sticky = N)
        self.tmpLabel2 = Label(self.egPanel, text = "")
        self.tmpLabel2.pack(side = TOP, pady = 5)
        self.egLabels = []
        for i in range(3):
            self.egLabels.append(Label(self.egPanel))
            self.egLabels[-1].pack(side = TOP)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight = 1)
        self.frame.rowconfigure(4, weight = 1)


    def loadDir(self, dbg = False):
        if not dbg:
            s = self.entry.get()
            self.parent.focus()
            self.category = str(s)
        else:
            s = r'D:\workspace\python\labelGUI'

        # get image list
        self.imageDir = os.path.join(r'./Images', self.category)
        self.imageList = glob.glob(os.path.join(self.imageDir, '*.png'))
        if len(self.imageList) == 0:
            print 'No .png images found in the specified dir!'
            return
        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

         # set up output dir
        self.outDir = os.path.join(r'./Labels', self.category)
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)
        self.loadImage()
        print '%d images loaded from %s' %(self.total, s)
    
    #get the rectangle's four corners
    def gRCorner(self,xc,yc,x0,y0): 
        w=abs(x0-xc)*2
        h=abs(y0-yc)*2
        x2=2*xc-x0
        y2=2*yc-y0
        x0,x2=max(x0,x2),min(x0,x2)
        y0,y2=max(y0,y2),min(y0,y2)
        corner_x=x0,x0,x2,x2
        corner_y=y0,y2,y2,y0
        return zip(corner_x, corner_y), w, h

    def loadImage(self):
        # load image
        imagepath = self.imageList[self.cur - 1]
        self.img = Image.open(imagepath)
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))

        # load labels
        self.clearBBox()
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        labelname = self.imagename + '.txt'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    if i == 0:
                        bbox_cnt = int(line.strip())
                        continue
                    tmp = [float(t.strip()) for t in line.split()]
                    # print tmp
                    self.bboxList.append(tuple(tmp))
                    xc,yc=tmp[0], tmp[1]
                    x0,y0=xc+tmp[2]/2,yc+tmp[3]/2
                    poly_tmp=list(self.gRCorner(xc,yc,x0,y0))
                    tmpId = self.mainPanel.create_polygon(poly_tmp[0], \
                                                            width = 2, \
                                                            outline = COLORS[(len(self.bboxList)-1) % len(COLORS)],\
                                                            fill='')
                    angle = cm.exp(m.radians(tmp[4])*1j)
                    offset = complex(xc, yc)
                    newxy=[]
                    for x, y in poly_tmp[0]:
                        v = angle * (complex(x, y) - offset) + offset
                        newxy.append(v.real)
                        newxy.append(v.imag)
                    # print np.angle(angle,deg=True)
                    self.mainPanel.coords(tmpId, *newxy)

                    self.bboxIdList.append(tmpId)
                    self.listbox.insert(END, '(%d, %d), w:%d, h:%d, deg:%.2f' %(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]))
                    self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])

    def saveImage(self):
        with open(self.labelfilename, 'w') as f:
            f.write('%d\n' %len(self.bboxList))
            for bbox in self.bboxList:
                f.write(' '.join(map(str, bbox)) + '\n')
        print 'Image No. %d saved' %(self.cur)

    def complex_unit(self,event):
        dx = self.mainPanel.canvasx(event.x) - self.STATE['x']
        dy = self.mainPanel.canvasy(event.y) - self.STATE['y']
        try:
            return complex(dx, dy) / abs(complex(dx, dy))
        except ZeroDivisionError:
            return 0.0 # cannot determine angle

    def mouseClick(self, event):
        # print "click state:{}".format(self.STATE['click']) 
        
        if self.STATE['click'] == 0:    
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        
        elif self.STATE['click']==1:
            xc, x0 = self.STATE['x'], event.x
            yc, y0 = self.STATE['y'], event.y
            self.STATE['gR'] = list(self.gRCorner(xc,yc,x0,y0))
            # print "Rectangle corner:",self.STATE['gR'][0]
            global start
            start =self.complex_unit(event)
        
        elif self.STATE['click']==2:
            self.bboxList.append((self.STATE['x'], self.STATE['y'], self.STATE['gR'][1], self.STATE['gR'][2],self.STATE['gR_deg']))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '(%d, %d), w:%d, h:%d, deg:%.2f' %(self.STATE['x'],self.STATE['y'], \
                                                                        self.STATE['gR'][1],self.STATE['gR'][2],\
                                                                        self.STATE['gR_deg']))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
            self.STATE['click'] = -1
        
        self.STATE['click'] = 1 + self.STATE['click']

    def mouseMove(self, event):
        self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))
        if self.tkimg: #mouse tracking
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width = 2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width = 2)
        
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            xc, x0 = self.STATE['x'], event.x
            yc, y0 = self.STATE['y'], event.y
            self.STATE['gR'] = list(self.gRCorner(xc,yc,x0,y0))
            # print self.STATE['gR']
            self.bboxId = self.mainPanel.create_polygon(self.STATE['gR'][0], \
                                                            width = 2, \
                                                            outline = COLORS[len(self.bboxList) % len(COLORS)],\
                                                            fill='')
        if 2 == self.STATE['click']:
            xc, xn = self.STATE['x'], event.x
            yc, yn = self.STATE['y'], event.y
            global start
            angle = self.complex_unit(event) / start
            offset = complex(xc, yc)
            newxy=[]
            for x, y in self.STATE['gR'][0]:
                v = angle * (complex(x, y) - offset) + offset
                newxy.append(v.real)
                newxy.append(v.imag)
            # print np.angle(angle,deg=True)
            self.STATE['gR_deg'] = np.angle(angle,deg=True)
            self.mainPanel.coords(self.bboxId, *newxy)


    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = -1

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1 :
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)
        self.STATE['click'] = 0

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.STATE['click'] = 0

    def prevImage(self, event = None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextImage(self, event = None):
        self.saveImage()
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()

if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.mainloop()
