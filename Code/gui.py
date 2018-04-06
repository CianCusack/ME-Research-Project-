from Tkinter import *
from record_race import record_race

# Run the recording software, until video input ends
def run(E1, E2, E3):
    record_race(int(E1.get()),E2.get(), E3.get())
    exit(1)

top = Tk()
top.grid()

# Add three inputs
L1 = Label(top, text="Estimated distance to buoy: ")
L1.pack( )
E1 = Entry(top, bd =5)
E1.pack()
L2 = Label(top, text="Colour of the buoy: ")
L2.pack( )
E2 = Entry(top, bd =5)
E2.pack()
L3 = Label(top, text="Direction of travel: ")
L3.pack( )
E3 = Entry(top, bd =5)
E3.pack()

# Create start button that invokes run()
B = Button(top, text ="Start race", command = lambda : run(E1, E2, E3))
B.pack(side=BOTTOM)

top.mainloop()