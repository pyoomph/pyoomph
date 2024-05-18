#!/usr/bin/env bash

mkdir -p _GATHER_SCRIPTS/pyoomph_tutorial_scripts
rm -rf _GATHER_SCRIPTS/pyoomph_tutorial_scripts/*

function gather()
{
 mkdir -p _GATHER_SCRIPTS/pyoomph_tutorial_scripts/$2
 cp $(find $1 -name *.py) _GATHER_SCRIPTS/pyoomph_tutorial_scripts/$2
}

gather temporal Temporal_ODEs
gather spatial Spatial_PDEs
gather pde SpatioTemporal_PDEs
gather ale Moving_Mesh
gather multidom Multiple_Domains
gather mcflow Multicomponent_Flow
gather dg Discontinuous_Galerkin
gather advstab Advanced_Linear_Dynamics
gather plotting Plotting_Interface

cat > _GATHER_SCRIPTS/pyoomph_tutorial_scripts/README.txt <<EOCAT
Here, you find the python scripts used and explained throughout the tutorial of pyoomph.
You can find pyoomph here: https://github.com/pyoomph/pyoomph
The tutorial is hosted here: https://pyoomph.readthedocs.io
EOCAT

(
cd _GATHER_SCRIPTS
rm -rf ../tutorial_example_scripts.zip
zip -r  ../tutorial_example_scripts.zip pyoomph_tutorial_scripts
)

rm -rf _GATHER_SCRIPTS
