# First time setup

Follow these steps.  You will only need to do this first part once.

You can paste into a terminal by pressing ctrl+shift+v .

Download and install miniconda from http://repo.continuum.io/miniconda/ by 
running the following commands in a terminal window:

```bash
wget http://repo.continuum.io/miniconda/Miniconda-3.0.0-Linux-x86_64.sh
bash Miniconda-3.0.0-Linux-x86_64.sh
```

- Say yes to the license.
- Use the default location for the install (just press enter).
- Say yes when it asks about prepending to your PATH.

Close the terminal and open a new one.

Run:

```bash
which conda
```

If conda is found then you did the previous steps correctly.  If you see:

```bash
/usr/bin/which: no conda in ( lots of stuff )
```

then you did something wrong and you need to fix it before continuing.  Once conda is properly installed you can delete the install file by running:

```bash
rm Miniconda-3.0.0-Linux-x86_64.sh
```

# Setting up the environment

Open a terminal and navigate to wherever you want to store your code.

Run the following commands:

```bash
git clone https://github.com/oxford-cs-advanced-ml-2014/practical1.git
cd practical1
./setup_environment.sh
```

This will build a local python environment with the dependencies needed for the 
practical.

Run the example code with:

```bash
./run.sh alternating_ridge_regression.py
```

This should output a bunch of numbers and then fail with a NameError.  If this 
happens then you have set up everything correctly.

If you move to a new machine you will need to run setup_environment.sh again.

You always need to run python code for the practicals through run.sh

# What next?

The instructions for the practical are available here: https://www.cs.ox.ac.uk/teaching/materials13-14/advml/practical1.pdf

