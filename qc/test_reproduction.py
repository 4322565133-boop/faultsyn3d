"""Numerical invariants for the reproduction, independent of gallery appearance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from su.geometry import rotation_matrix,dip_strike_from_normal,bezier_control_point
from su.hull_constraint import _disjoint_mask
from su.topology import tree_from_spec,VALID_EDGES,build_tree,validate_edge
from su.optimize import objective,objective_batch,decode
from su.reproduction import surface_graph,validate_surfaces,_triangles_disjoint,RejectedSample
from su.displacement import drag_operator


class ReproductionInvariants(unittest.TestCase):
    def test_control_axis_roundoff_tie(self):
        p0=np.array([-87.,-83.,0.]);p2=np.array([87.,-83.,0.])
        expected=bezier_control_point(p0,p2,1.,.1)
        changed=bezier_control_point(p0+np.array([0,2e-14,-2e-14]),p2,1.,.1)
        np.testing.assert_allclose(changed,expected,atol=1e-10)

    def test_frame_rotation_invariance(self):
        rng=np.random.default_rng(81)
        for _ in range(500):
            r=rotation_matrix(rng.uniform(0,360),rng.uniform(.01,179.99))
            d,s=dip_strike_from_normal(r[2]);restored=rotation_matrix(s,d)
            self.assertAlmostEqual(abs(restored[2]@r[2]),1.,places=12)
            np.testing.assert_allclose(restored@restored.T,np.eye(3),atol=1e-12)

    def test_contained_triangle_is_not_disjoint(self):
        cube=np.array([[x,y,z] for x in [-10.,10.] for y in [-10.,10.] for z in [-10.,10.]])
        result=_disjoint_mask(np.array([-1.,0,0]),np.array([1.,0,0]),np.array([1.]),np.array([.5]),cube,0)
        self.assertFalse(result[0,0])

    def test_triangle_sat(self):
        a=np.array([[0.,0],[1.,0],[0.,1]])
        self.assertFalse(_triangles_disjoint(a,a*.1+.1))
        self.assertTrue(_triangles_disjoint(a,a+3))
        self.assertFalse(_triangles_disjoint(a,a))
        line=np.array([[0.,0],[.5,0],[1,0]])
        self.assertTrue(_triangles_disjoint(line,line+[0,1]))
        self.assertFalse(_triangles_disjoint(line,line+[.25,0]))

    def test_P_forest_counterexample(self):
        tree=tree_from_spec([(0,('P','P')),(0,('P','P'))])
        z=np.array([[0.,0,0],[1,3,1],[2,1,2]])
        self.assertGreater(objective(tree,z,3),0)
        f=tree.direction_forest(1)
        self.assertEqual(len(set(f.components.values())),3)

    def test_cross_component_P_not_waived_by_Y(self):
        tree=tree_from_spec([(0,('X','P')),(1,('Y','Y'))])
        z=np.array([[0.,0,0],[1,-6,7],[11,-20,-6]])
        self.assertGreater(objective(tree,z,3),0)

    def test_vectorised_penalty_matches_scalar(self):
        rng=np.random.default_rng(6)
        for _ in range(20):
            tree=build_tree([VALID_EDGES[i] for i in rng.integers(0,7,6)],3,rng)
            z=np.stack([decode(rng.uniform(-10,10,3*(len(tree)-1)),len(tree),np.array([-10]*3),np.array([10]*3)) for _ in range(35)])
            np.testing.assert_allclose(objective_batch(tree,z),[objective(tree,a,len(tree)) for a in z])

    def test_bezier_preserves_reference_endpoints(self):
        a=np.array([[-60.,-60,4],[60,-60,20],[-60,60,-5]])
        xs=ys=np.linspace(-60,60,129)
        h=surface_graph(a,xs,ys,.1,.15)
        np.testing.assert_allclose([h[0,0],h[-1,0],h[0,-1],h[-1,-1]],[4,20,-5,11],atol=1e-10)

    def test_seven_edge_geometries(self):
        endpoints={'P':3.,'X':-3.,'Y':-3.}
        xs=ys=np.linspace(-60,60,129);R=rotation_matrix(90,90)
        for edge in VALID_EDGES:
            tree=tree_from_spec([(0,edge)])
            z=np.array([[0.,0,0],[3,endpoints[edge[0]],endpoints[edge[1]]]])
            self.assertEqual(objective(tree,z,2),0)
            anchors=np.array([[-60.,-60,0],[60,-60,0],[-60,60,0]])
            hs=[]
            for zi in z:
                ai=anchors.copy();ai[:,2]=zi*4
                hs.append(surface_graph(ai,xs,ys,0.,0.))
            valid,inside,checks=validate_surfaces(tree,np.array(hs),xs,ys,np.array([64]*3),R,(128,128,128),z)
            self.assertEqual(len(checks),1)
            if 'Y' in edge:
                self.assertTrue(np.any(~valid[1]))
                self.assertTrue(np.all((hs[1]-hs[0])[valid[1]]<=1e-8))

    def test_drag_one_sided_limits_and_far_boundary(self):
        for mode,sign in [('normal',1),('reverse',-1)]:
            vals=drag_operator(np.array([-1e-8,1e-8]),mode)
            np.testing.assert_allclose(vals,[-sign,sign],atol=1e-6)
            np.testing.assert_allclose(drag_operator(np.array([-1.,1.]),mode),[0,0],atol=1e-12)
        self.assertGreater(drag_operator(np.array([.15]),'normal')[0],1)


if __name__=='__main__':unittest.main(verbosity=2)
