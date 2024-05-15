#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2024  Christian Diddens & Duarte Rocha
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
#
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================
 
from ..generic.problem import Problem
from .mesh import MeshFromTemplate2d
import numpy
from ..typings import *
from scipy.spatial import Delaunay

class PFEMMeshUpdater:
    def __init__(self,problem:Problem,meshname:str) -> None:
        super().__init__()
        self.problem=problem
        self.meshname=meshname

    def get_mesh(self)->MeshFromTemplate2d:
        mesh=self.problem.get_mesh(self.meshname)
        assert isinstance(mesh,MeshFromTemplate2d)
        return mesh

    def alpha_shape(self,tri, alpha, only_outer=True):
        """
        Compute the alpha shape (concave hull) of a set of points.
        :param points: np.array of shape (n,2) points.
        :param alpha: alpha value.
        :param only_outer: boolean value to specify if we keep only the outer border
        or also inner edges.
        :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
        """

        def add_edge(edges, i, j):
            """
            Add an edge between the i-th and j-th points,
            if not in the list already
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                assert (j, i) in edges, "Can't go twice over same directed edge right?"
                if only_outer:
                    # if both neighboring triangles are in shape, it's not a boundary edge
                    edges.remove((j, i))
                return
            edges.add((i, j))

        innersimplices:List[List[int]]=[]
        edges = set()
        # Loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.simplices:
            pa = tri.points[ia]
            pb = tri.points[ib]
            pc = tri.points[ic]
            # Computing radius of triangle circumcircle
            # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
            a = numpy.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = numpy.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = numpy.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
            s = (a + b + c) / 2.0
            area = numpy.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            if circum_r < alpha:
                add_edge(edges, ia, ib)
                add_edge(edges, ib, ic)
                add_edge(edges, ic, ia)
                innersimplices.append([ia,ib,ic])
        return edges,innersimplices
    
    def delaunay_with_alpha_shape(self,coords:List[Tuple[float,float]],alpha:float):
        tri=Delaunay(coords)
        edges,innersimplices=self.alpha_shape(tri,0.15)        
        dom=self.get_mesh()
        # Create a new mesh
        for s in innersimplices:
            n1=dom.node_pt(s[0])
            n2=dom.node_pt(s[1])
            n3=dom.node_pt(s[2])
            dom.add_tri_C1(n1,n2,n3)        
        return edges

    def define_new_mesh(self,coords:List[Tuple[float,float]]):
        raise RuntimeError("Implement this here")

    def update_mesh(self):
        dom=self.get_mesh()
        coords=[(n.x(0),n.x(1),) for n in dom.nodes()]
        # Remove interfaces and remove the nodes from the boundary       
        for n in dom.nodes():
            binds=n.get_boundary_indices()
            for b in binds:
                n.remove_from_boundary(b)
        for inam,imesh in dom._interfacemeshes.items():
            imesh.clear_before_adapt()
        for inam,imesh in dom._interfacemeshes.items():
            for inam2,imesh2 in imesh._interfacemeshes.items():
                imesh2.clear_before_adapt()
        for inam,imesh in dom._interfacemeshes.items():
            for inam2,imesh2 in imesh._interfacemeshes.items():
                for inam3,imesh3 in imesh2._interfacemeshes.items():
                    imesh3.clear_before_adapt()
        

        dom.flush_element_storage()        
        self.define_new_mesh(coords)            

        dom.recreate_boundary_information()

        for inam,imesh in dom._interfacemeshes.items():
            imesh.rebuild_after_adapt()
        for inam,imesh in dom._interfacemeshes.items():
            for inam2,imesh2 in imesh._interfacemeshes.items():
                imesh2.rebuild_after_adapt()
        for inam,imesh in dom._interfacemeshes.items():
            for inam2,imesh2 in imesh._interfacemeshes.items():
                for inam3,imesh3 in imesh2._interfacemeshes.items():
                    imesh3.rebuild_after_adapt()                
        dom.get_problem().rebuild_global_mesh()
        dom.get_problem().reapply_boundary_conditions()
        dom.get_problem().invalidate_cached_mesh_data()
        dom.get_problem().invalidate_eigendata()

