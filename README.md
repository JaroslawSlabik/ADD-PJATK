# ADD-PJATK
Internet ad filter - project for classes Big Data Analysis - PJATK

## INFO
For the code to work properly, Python version 3.8 should be used.
The requirements.txt file contains the required libraries.
There is input in the ad.data file.
The program code is in the proj.py file.
It is recommended to use a virtual environment and library manager.
While running, the program generates PNG files with charts.

## INSTALLATION AND START-UP
Installing python 3.8

    $ sudo apt-get install python3.8

Installing the library manager

    $ sudo apt-get install python3-pip

Installation of a virtual environment (optional)

    $ pip3 install virtualenv

Create a virtual environment (optional)

    $ virtualenv -p /usr/bin/python3.8 venv

Activation of the virtual environment (optional)

    $ source venv/bin/activate

Installation of the required libraries

    $ pip3 install -r requirements.txt

Running the program in the non-standardized mode

    $ python3.8 proj.py false

Running the program in the standardized mode

    $ python3.8 proj.py true
