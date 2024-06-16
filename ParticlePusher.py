import numpy as np
import matplotlib.pyplot as plt

def advance_particle_v(v, rqm, E, B, dt):
    """
    Advances particle velocity 'v' (non-relativistic algorithm).
    This is the so-called "Boris" algorithm to compute the change in the particle velocity due to the Lorentz force.
    The algoritm takes in current particle velocity (v), the particle charge to mass ratio (rqm), the instantaneous electric (E) and magnetic (B) fields felt by the particle, and the time step of integration (dt).
    The output of the algorithm is to advance the particle velocity (v) by one time step.
    
    Details of the algorithm can be found in the reference below.
    
    Reference:
    J. P. Boris, Relativistic plasma simulation-optimization of a hybrid code, Proc. Fourth Conf. Num. Sim. Plasmas, Naval Res. Lab., Wash., D. C., 3-67, 2-3 November 1970
    """ 
    
    vminus = v + 0.5*rqm*E*dt
    
    t = 0.5*rqm*B*dt
    s = 2*t/(1.+ np.sum(t**2))
    
    vprime = np.zeros(3)
    vplus = np.zeros(3)
    
    vprime[0] = vminus[0] + vminus[1]*t[2] - vminus[2]*t[1]
    vprime[1] = vminus[1] + vminus[2]*t[0] - vminus[0]*t[2]
    vprime[2] = vminus[2] + vminus[0]*t[1] - vminus[1]*t[0]
    
    vplus[0] = vminus[0] + vprime[1]*s[2] - vprime[2]*s[1]
    vplus[1] = vminus[1] + vprime[2]*s[0] - vprime[0]*s[2]
    vplus[2] = vminus[2] + vprime[0]*s[1] - vprime[1]*s[0]
    
    v = vplus + 0.5*rqm*E*dt
    
    return v

def advance_particle_x(x, v, dt):
    """
    This routine advances the particle position 'x' by one time step.
    """ 
    x = x + v*dt
    return x


def relativistic_advance_particle_u(u, rqm, E, B, dt):
    """
    This is the so-called "Boris" algorithm to compute the change in the relativistic particle momentum due to the Lorentz force.
    The algoritm takes in current particle proper velocity (u = gamma*v, where gamma is the Lorentz factor and v is the particle speed), the particle charge to mass ratio (rqm), the instantaneous electric (E) and magnetic (B) fields felt by the particle, and the time step of integration (dt).
    The output of the algorithm is to advance the particle proper velocity (u) by one time step.
    
    Details of the algorithm can be found in the reference below.
    
    Reference:
    J. P. Boris, Relativistic plasma simulation-optimization of a hybrid code, Proc. Fourth Conf. Num. Sim. Plasmas, Naval Res. Lab., Wash., D. C., 3-67, 2-3 November 1970
    """ 
    
    uminus = u + 0.5*rqm*E*dt
    gamma = np.sqrt(1.+np.sum(uminus**2))
    
    t = 0.5*rqm*B*dt/gamma
    s = 2*t/(1.+ np.sum(t**2))
    
    uprime = np.zeros(3)
    uplus = np.zeros(3)
    
    uprime[0] = uminus[0] + uminus[1]*t[2] - uminus[2]*t[1]
    uprime[1] = uminus[1] + uminus[2]*t[0] - uminus[0]*t[2]
    uprime[2] = uminus[2] + uminus[0]*t[1] - uminus[1]*t[0]
    
    uplus[0] = uminus[0] + uprime[1]*s[2] - uprime[2]*s[1]
    uplus[1] = uminus[1] + uprime[2]*s[0] - uprime[0]*s[2]
    uplus[2] = uminus[2] + uprime[0]*s[1] - uprime[1]*s[0]
    
    u = uplus + 0.5*rqm*E*dt
    
    return u

def relativistic_advance_particle_x(x, u, dt):
    gamma = np.sqrt(1.+np.sum(u**2))
    v = u/gamma
    x = x + v*dt
    return x

def EM_wave(x, t, omega, a0):
    # Linearly polarized EM wave, polarization along x-axis (eps=0)
    return EM_wave_polarized(x, t, omega, a0, eps=0)

def EM_wave_polarized(x, t, omega, a0, eps):
    
    z = x[-1]
    E0 = a0*omega
    k  = omega
    
    tau = 8*np.pi
    z0 = 4*tau
    tenv = np.exp(-(z+z0-t)**2/(2*tau**2)) # temporal envelope (to model a pulse of duration tau)

    Ex = 1/np.sqrt(1 + eps * eps) * E0 * np.sin(k * z - omega*t) * tenv
    Ey = eps/np.sqrt(1 + eps * eps) * E0 * np.cos(k * z - omega*t) * tenv
    Bx = -Ey
    By = Ex
    return [Ex, Ey, 0], [Bx, By, 0]

def integrate_particle_trajectory_elliptical(u0 = [0.0, 0.0, 0.0], x0 = [0, 0.0, 0.0], rqm = 1, dt = 0.1, T = 3*2*np.pi, a0 = 0.01, eps = 1):
    
    #This given calculates the trajectory of a particle with charge to mass ratio 'rqm' with initial conditions given by:
    #v0 = [vx0, vy0, vz0]
    #x0 = [x0, y0, z0]
    #
    #The electric and magnetic field profiles must be specified in the routines B_profile(x,t) and E_profile(x,t).
    #The trajectory is integrated for upto time T (in units of 1/wc, where wc = eB0/mc)
    #The output of this routine is:
    #- t (a 1D time array of size nsteps = T/dt)
    #- x (a 2D array of size (nsteps,3), which contains the x,y,z position of the particle for each time step)
    #- v (a 2D array of size (nsteps,3), which contains the vx,vy,vz velocity components of the particle for each time step)
    

    nsteps = int(T/dt) # number of time steps

    u = np.zeros((nsteps, 3))
    x = np.zeros((nsteps, 3))
    B = np.zeros((nsteps, 3))
    E = np.zeros((nsteps, 3))

    u[0,:] = u0
    x[0,:] = x0
    
    E[0,:], B[0,:] = EM_wave(x = x[0,:], t = 0, omega = 1, a0 = a0)
    
    t = np.arange(nsteps)*dt
    for n in range(1,nsteps):
        u[n,:] = relativistic_advance_particle_u(u = u[n-1,:], rqm = rqm, E = E[n-1,:], B = B[n-1,:], dt = dt) # advance particle velocity according to Lorentz force (using the Boris algorithm; see particle_pusher.py for more details.)
        x[n,:] = relativistic_advance_particle_x(x = x[n-1,:], u = u[n,:], dt = dt) # advance particle position according to new particle velocity
        E[n,:], B[n,:] = EM_wave_polarized(x = x[n,:], t = t[n], omega = 1, a0 = a0, eps = eps)

    
    return t, x, u

t, x, u = integrate_particle_trajectory_elliptical(u0 = [0, 0.0, 0.0], 
                                        x0 = [0, 0.0, 0.0], 
                                        rqm = 1, dt = 0.05, T = 40*2*np.pi, 
                                        a0 = 0.1, eps = 1)

u_abs2 = np.sum(u**2, axis=1)
gamma = np.sqrt(1.+u_abs2)
v = u/gamma[:, np.newaxis]
plt.figure()
plt.title('Particle track')
plt.plot(x[:,0], x[:,1])
plt.xlabel(r'$x [v_0/\omega_c]$')
plt.ylabel(r'$y [v_0/\omega_c]$')

plt.figure()
plt.title('CP, a0 = 2')
plt.plot(t[:], v[:,0], label="$v_x$")
plt.plot(t[:], v[:,1], label="$v_y$")
plt.plot(t[:], v[:,2], label="$v_z$")
plt.plot(t[:], np.sqrt(v[:,0]**2 + v[:,1]**2+ v[:,2]**2), label="$|v|$")
plt.xlabel(r'$t [\omega_0^{-1}]$')
plt.ylabel(r'$v/c$')
plt.legend()
plt.show()