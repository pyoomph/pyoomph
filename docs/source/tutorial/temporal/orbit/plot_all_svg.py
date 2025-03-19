import warnings

warnings.filterwarnings("error")

import matplotlib.pyplot as plt
import numpy ,glob

ax = plt.figure().add_subplot(projection='3d')
triv_stab=numpy.loadtxt("lorenz_trivial/stable_branch.txt")
pitch1_stab=numpy.loadtxt("lorenz_pitchfork/stable_branch1.txt")
pitch2_stab=numpy.loadtxt("lorenz_pitchfork/stable_branch2.txt")
pitch1_unstab=numpy.loadtxt("lorenz_pitchfork/unstable_branch1.txt")
pitch2_unstab=numpy.loadtxt("lorenz_pitchfork/unstable_branch2.txt")

selected_orbit=numpy.loadtxt("lorenz_unstable_at_14/orbit_at_rho_16.9880/lorenz.txt")
traj_orbit_unstable=numpy.loadtxt("lorenz_unstable_at_14/lorenz.txt")
chaos=numpy.loadtxt("lorenz_pitchfork_leave/lorenz.txt")
import mpl_toolkits.mplot3d.art3d as art3d




# Create the 3D-line collection object

minrho=0
maxrho=30
cmap=plt.get_cmap('seismic')
norm=plt.Normalize(minrho, maxrho)
def colline(data,stable,rho_override=None):    
    rho,x,y,z=data[:,0],data[:,1],data[:,2],data[:,3]
    points = numpy.array([x, y, z]).T.reshape(-1, 1, 3)
    if rho_override is None:
        segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
        lc = art3d.Line3DCollection(segments, cmap=cmap,norm=norm)    
        lc.set_array(rho) 
        if stable:
            lc.set_linewidth(2)
        else:
            lc.set_linewidth(0.5)
        ax.add_collection3d(lc, zs=z, zdir='z')
    else:
        ax.plot(x,y,zs=z,zdir='z',color=cmap(norm(rho_override)),linewidth=2 if stable else 0.5)
    
    
for i in range(30,31):        
    ax=plt.figure().add_subplot(projection='3d')        
    colline(pitch1_stab,stable=True)
    colline(pitch2_stab,stable=True)
    if i>=1:
        colline(pitch1_unstab,stable=False)
        colline(pitch2_unstab,stable=False)
        if i>=2:
            survided_loop=True
            numorbits=0
            if i<2000:
                for dn in reversed(sorted(glob.glob("lorenz_hopf_orbits/orbit_at_rho_*"))):
                    rho=float(dn.split("_")[-1].split(".")[0])
                    if rho>14:
                        data=numpy.loadtxt(dn+"/lorenz.txt")    
                        colline(data,stable=False,rho_override=rho)
                        data[:,1]*=-1
                        data[:,2]*=-1
                        colline(data,stable=False,rho_override=rho)
                        numorbits+=1
                        if i<2+numorbits:
                            survided_loop=False
                            break

            if survided_loop:
                j=i-numorbits-2                
                if i<900:
                    colline(selected_orbit,stable=j>0,rho_override=16.9880)
                    if j>0:                
                        k=min(3*(j-1),len(traj_orbit_unstable))
                        ax.plot(traj_orbit_unstable[:k,1],traj_orbit_unstable[:k,2],zs=traj_orbit_unstable[:k,3],zdir="z",color="black",linewidth=0.5)
                        ax.scatter(traj_orbit_unstable[k-1,1],traj_orbit_unstable[k-1,2],zs=traj_orbit_unstable[k-1,3],zdir="z",color="black",s=10)
                        kr=(k)%(len(selected_orbit)-1)
                        ax.scatter(selected_orbit[kr,1],selected_orbit[kr,2],zs=selected_orbit[kr,3],zdir="z",color="black",s=10,marker="x")
                else:
                    j=i-900
                    k=min(10*j,len(chaos))
                    ax.plot(chaos[:k,1],chaos[:k,2],zs=chaos[:k,3],zdir="z",color="black",linewidth=0.5)
                    ax.scatter(chaos[k-1,1],chaos[k-1,2],zs=chaos[k-1,3],zdir="z",color="black",s=10)                    
                    if i>2000:
                        cnt=0
                        for dn in reversed(sorted(glob.glob("lorenz_unstable_orbit_fourier/orbit_at_rho_*.txt"))):                            
                            rho=float(dn.split("_")[-1].split(".")[0])
                            unstable_in_chaos=numpy.loadtxt(dn+"/lorenz.txt")                            
                            colline(unstable_in_chaos,stable=False,rho_override=rho)
                            cnt+=1
                            if cnt>i-2000:
                                break
                    

    #ax.legend()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 30)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=20., azim=2*35+0.5*i, roll=0)


    plt.tight_layout()
    #äplt.show()
    plt.savefig("svgs/plot_{:04d}.svg".format(i))
    plt.close()