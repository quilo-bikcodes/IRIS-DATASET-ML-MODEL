from tkinter import *
import tkinter as tk
import tkinter as ttk
from functions import pltgraph
import pandas as pd


# Loading the Data Set
iris_df = pd.read_csv('./data/iris.csv')
dataplt = pltgraph(iris_df)
def showplt():
    dataplt.get_linepltSLcm()
def showstats():   
    dataplt.get_stats()


LARGEFONT =("Verdana", 35)
  
class tkinterApp(tk.Tk):
     
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
         
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Iris Dataset Analysis")
        self.geometry("500x200")
         
        # creating a container
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        # initializing frames to an empty array
        self.frames = {} 
  
        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, Page1, DataStatsPage,PltSelectPage,LinePltPage,ViolinPltPage):
  
            frame = F(container, self)
  
            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
  
# first window frame startpage
  
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
         
        label=Label(self,text="WELCOME TO IRIS DATASET ML MODEL")
        button1 = Button(self,text ="NEXT",width=10,command=lambda : controller.show_frame(Page1))
        button2 = Button(self,text="QUIT",width=10,command=self.destroy)
        #Display label
        label.place(relx=.5, rely=.3,anchor= CENTER)
        button1.place(relx=.4, rely=.5,anchor= CENTER)
        button2.place(relx=.6, rely=.5,anchor= CENTER)

  
          
  
  
# second window frame page1
class Page1(tk.Frame):
     
    def __init__(self, parent, controller):
         
        tk.Frame.__init__(self, parent)
        label1 = Label(self,text="Choose Your Requirements")
        button1 = Button(self,text="Data Plots",command=lambda : controller.show_frame(PltSelectPage))
        button2 = Button(self,text="Data Stats",command=lambda : controller.show_frame(DataStatsPage))
        button3 = Button(self,text="ML Prediction")

        label1.place(relx=.5, rely=.3,anchor= CENTER)
        button1.place(relx=.3, rely=.5,anchor= CENTER)
        button2.place(relx=.5, rely=.5,anchor= CENTER)
        button3.place(relx=.7, rely=.5,anchor= CENTER)



  
  
  
  
# third window frame page2
class DataStatsPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        stats = dataplt.get_stats()
        label1 = Label(self,text=stats)
        button1 = Button(self,text="BACK",command=lambda : controller.show_frame(Page1))

        label1.place(relx=.5, rely=.6,anchor= CENTER)
        button1.place(relx=.5, rely=0.1,anchor= CENTER)
class PltSelectPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label1 = Label(self,text="Choose Your Requirements")
        button1 = Button(self,text="Line Plots",command= lambda: controller.show_frame(LinePltPage))
        button2 = Button(self,text="Violin Plots",command=lambda : controller.show_frame(ViolinPltPage))
        button3 = Button(self,text="Pair Plot",command=dataplt.get_pairplt)
        bckbtn = Button(self,text="BACK",command=lambda: controller.show_frame(Page1))


        label1.place(relx=.5, rely=.3,anchor= CENTER)
        button1.place(relx=.3, rely=.5,anchor= CENTER)
        button2.place(relx=.5, rely=.5,anchor= CENTER)
        button3.place(relx=.7, rely=.5,anchor= CENTER)
        bckbtn.place(relx=.5, rely=.7,anchor= CENTER)
class LinePltPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label1 = Label(self,text="Line Plots")
        button1 = Button(self,text="Sepal Length",command=dataplt.get_linepltSLcm)
        button2 = Button(self,text="Sepal Width",command=dataplt.get_linepltSWcm)
        button3 = Button(self,text="Petal Length",command=dataplt.get_linepltPLcm)
        button4 = Button(self,text="Petal Width",command=dataplt.get_linepltPWcm)
        bckbtn = Button(self,text="BACK",command=lambda: controller.show_frame(PltSelectPage))

        label1.place(relx=.5, rely=.2,anchor= CENTER)
        button1.place(relx=.4, rely=.4,anchor= CENTER)
        button2.place(relx=.6, rely=.4,anchor= CENTER)
        button3.place(relx=.4, rely=.6,anchor= CENTER)
        button4.place(relx=.6, rely=.6,anchor= CENTER)
        bckbtn.place(relx=.5, rely=.8,anchor= CENTER)

class ViolinPltPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label1 = Label(self,text="Violin Plots")
        button1 = Button(self,text="Sepal Length",command=dataplt.get_violinpltSLcm)
        button2 = Button(self,text="Sepal Width",command=dataplt.get_violinpltSWcm)
        button3 = Button(self,text="Petal Length",command=dataplt.get_violinpltPLcm)
        button4 = Button(self,text="Petal Width",command=dataplt.get_violinpltPWcm)
        bckbtn = Button(self,text="BACK",command=lambda: controller.show_frame(PltSelectPage))

        label1.place(relx=.5, rely=.2,anchor= CENTER)
        button1.place(relx=.4, rely=.4,anchor= CENTER)
        button2.place(relx=.6, rely=.4,anchor= CENTER)
        button3.place(relx=.4, rely=.6,anchor= CENTER)
        button4.place(relx=.6, rely=.6,anchor= CENTER)
        bckbtn.place(relx=.5, rely=.8,anchor= CENTER)

    


       
  
#  lambda : controller.show_frame(StartPage)
# Driver Code
app = tkinterApp()
app.mainloop()
