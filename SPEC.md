We are going to create a repo for drawing with electric fields and conductors. 

The bread and  butter functionality is going to be as follows: ghe user
specifies a canvas size and imports masks as pngs that act as conductors (alpha
channel delineates conductor). Voltages are specified on the boundary of the
canvas and on the conductor masks. Then the Poisson equation is solved to get
the electric field.  Then we do a high quality line integral convolution to
display the result. The intention is to make a "drawing program" where the
primitives are conductors and line integral convolutions.

However, some of the effort will go into making various conveniences,
good post-processing, easy ability to manipulate conductors and color schemes etc. 
We are going to focus on writing good proceedural code so that the codebase is easy to extend.
In the longer run we will add a gui, but we will first add the bread an butter.

Rough initial thoughts on what we need

mask_utilities: for "cutting" out interiors of pngs to create thin metal shells
poisson_solve: just what it sounds like. very very important that it is performant. Should reuse libraries if possible
lic: thin wrapper around rLIC
post_processing: tbd 
color_canvas: creation of noise canvas or alternatives that will be smeared 
