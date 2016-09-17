#from math import *
import numpy as np
import math
try:
    import psyco
    psyco.full()
except ImportError:
    pass

R_earth = 6371000.785

sin = np.sin
cos = np.cos
tan = np.tan
atan = np.arctan
acos = np.arccos
asin = np.arcsin
log = np.log
exp = np.exp
sqrt = np.sqrt
pi = math.pi
radians = np.radians
degrees = np.degrees

def UTMReverse(E, N, lmb0, FN = 0.0 ): #328.):
    # parameters
    M0 = 0. # natural origin north
    FE = 500000. # false eastening
    # constants
    k0 = 0.9996
    # ellipsoid
    a = 6377563.396
    #oneDivF = 299.32496
    eSqr = 0.00667054

    eSSqr = eSqr/(1 - eSqr)
    e1 = (1 - (1 - eSqr)**0.5)/(1 + (1 + eSqr)**0.5)
    M1 = M0 + (N - FN)/k0
    mu1 = M1/(a*(1 - eSqr/4 - 3*(eSqr**2)/64 - 5*(eSqr**3)/256))
    phi1 = mu1 + (3 *(e1**1)/2  - 27*(e1**3)/32) * sin(2*mu1) + \
                 (21*(e1**2)/16 - 55*(e1**4)/32) * sin(4*mu1) + \
                 (151*(e1**3)/96               ) * sin(6*mu1) + \
                 (1097*(e1**4)/512             ) * sin(8*mu1)
    T1 = tan(phi1)**2
    C1 = eSSqr * (cos(phi1)**2)
    nu1 = a / ((1 - eSqr*(sin(phi1)**2))**0.5)
    ro1 = a*(1-eSqr)/((1 - eSqr*(sin(phi1)**2))**1.5)
    D = (E - FE)/(nu1 * k0)
    
    phi = phi1 - (nu1 * tan(phi1) / ro1) * ((D**2)/2 - \
                  (5 + 3*T1 + 10*C1 - 4*(C1**2) - 9*eSSqr) * (D**4) / 24 + \
                  (61 + 90*T1 + 298*C1 + 45*(T1**2) - 252 * eSSqr - 3 * (C1**2)) * (D**6)/720)
    lmb = lmb0 + (D - (1 + 2*T1 + C1)*(D**3)/6 + \
                  (5 - 2*C1 + 28*T1 - 3*(C1**2) + 8*eSSqr + 24*(T1**2))*(D**5)/120)/cos(phi1)
    return (lmb, phi)

def UTMForward(lmb, phi, lmb0):
    # parameters
    M0 = 0. # natural origin north
    FN = 0. # false northing
    FE = 500000. # false eastening
    # constants
    k0 = 0.9996
    # ellipsoid
    a = 6377563.396
    #oneDivF = 299.32496
    eSqr = 0.00667054

    eSSqr = eSqr/(1 - eSqr)
    T = tan(phi)**2
    C = eSqr * (cos(phi)**2) / (1 - eSqr)
    A = (lmb - lmb0) * cos(phi)
    nu = a / ((1 - eSqr*(sin(phi)**2))**0.5)
    M = a * ((1 - eSqr/4 - 3 * (eSqr**2)/64 - 5 * (eSqr**3)/256) * phi - \
             (3*eSqr/8 + 3*(eSqr**2)/32 + 45*(eSqr**3)/1024) * sin(2*phi) + \
             (15*(eSqr**2)/256 + 45*(eSqr**3)/1024) * sin(4*phi) + \
             (35*(eSqr**3)/3072) * sin(6*phi))
    E = FE + k0*nu*(A + (1 - T + C)*(A**3)/6 + \
                    (5 - 18*T + T**2 + 72*C - 58*eSSqr)*(A**5)/120)
    N = FN + k0*(M - M0 + nu * tan(phi)*((A**2)/2 + \
                    (5 - T + 9*C + 4*(C**2))*(A**4)/24 + \
                    (61 - 58*T + T**2 + 600*C - 330*eSSqr)*(A**6)/720))
    return (E,N)

def TopocentricForward(lmb, phi, lmb0, phi0):
    # ellipsoid
    a = 6377563.396
    #oneDivF = 299.32496
    eSqr = 0.00667054
    
    h = 0.0 # height above ground, approx.
    # convert to radians
    lmb = radians(lmb)
    phi = radians(phi)
    lmb0 = radians(lmb0)
    phi0 = radians(phi0)
    # precompute sin/cos
    sinlmbMlmb0 = sin(lmb-lmb0)
    coslmbMlmb0 = cos(lmb-lmb0)
    sinphi = sin(phi)
    cosphi = cos(phi)
    sinphi0 = sin(phi0)
    cosphi0 = cos(phi0)
    
    nu  = a / ((1. - eSqr*(sinphi **2))**0.5)
    nu0 = a / ((1. - eSqr*(sinphi0**2))**0.5)
    
    U = (nu + h)*cosphi * sinlmbMlmb0
    V = (nu + h)*(sinphi*cosphi0 - cosphi*sinphi0*coslmbMlmb0) + eSqr*(nu0*sinphi0 - nu*sinphi)*cosphi0
    return (U,V)

def TopocentricBackward(U,V,lmb0,phi0):
    # ellipsoid
    a = 6377563.396
    oneDivF = 299.32496
    eSqr = 0.00667054
    
    W = 0.0
    h0 = 0.0 #?
    # convert to radians
    lmb0 = radians(lmb0)
    phi0 = radians(phi0)
    # precompute sin/cos
    sinphi0 = sin(phi0)
    cosphi0 = cos(phi0)
    sinlmb0 = sin(lmb0)
    coslmb0 = cos(lmb0)
    
    nu0 = a / ((1. - eSqr*(sinphi0**2))**0.5)
    
    X0 = (nu0 + h0)*cosphi0*coslmb0
    Y0 = (nu0 + h0)*cosphi0*sinlmb0
    Z0 = ((1-eSqr)*nu0 + h0)*sinphi0
    
    X = X0 - U*sinlmb0 - V*sinphi0*coslmb0 + W*cosphi0*coslmb0
    Y = Y0 + U*coslmb0 - V*sinphi0*sinlmb0 + W*cosphi0*sinlmb0
    Z = Z0 + V*cosphi0 + W*sinphi0
    
    eps = eSqr/(1-eSqr)
    b = a*(1 - oneDivF)
    p = (X**2 + Y**2)**0.5
    q = atan((Z*a)/(p*b))
    phi = atan((Z+eps*b*(sin(q)**3))/(p - eps*a*(cos(q)**3)))
    lmb = atan(Y/X)
    return degrees(phi), degrees(lmb)
               

def UTMZoneForward(lmb, phi, zone):
    lmb0 = radians(zone * 6 - (180 + 6/2))
    return UTMForward(lmb, phi, lmb0)

def UTMZoneReverse(E, N, zone, FN = 0.0):
    lmb0 = radians(zone * 6 - (180 + 6/2))
    return UTMReverse(E, N, lmb0, FN)

def SwissForward(lmb, phi):
    #a = 6377397.155
    EE = 0.006674372230614
    E = sqrt(EE)
    #phi0 = radians(46.0 + 57./60. + (8.66)/3600.)
    lambda0 = radians(7.0 + 26./60. + (22.50)/3600.)
    R =  6378815.90365
    alpha =  1.00072913843038
    b0 = radians(46.0 + 54./60. + (27.83324844/3600.))
    K = 0.0030667323772751
    S = -alpha * log(tan(pi/4. - phi/2)) - alpha*E/2 * log( (1+E*sin(phi))/(1-E*sin(phi)) ) + K
    b = 2*(atan(exp(S)) - pi/4)
    l = alpha * (lmb - lambda0)
    l_ = atan( sin(l) / (sin(b0)*tan(b) + cos(b0)*cos(l)) )
    b_ = asin( cos(b0)*sin(b) - sin(b0)*cos(b)*cos(l) )
    Y = R*l_
    X = R/2 * log( (1+sin(b_))/(1-sin(b_)) )
    return X+200000, Y+600000

# no idea what went wrong in the swiss transformations ...
def SwissReverse(yy,  xx):
    #a = 6377397.155
    #EE = 0.006674372230614
    #E = sqrt(EE)
    #phi0 = radians(46.0 + 57./60. + (8.66)/3600.)
    #lambda0 = radians(7.0 + 26./60. + (22.50)/3600.)
    #R =  6378815.90365
    #alpha =  1.00072913843038
    #b0 = radians(46.0 + 54./60. + (27.83324844/3600.))
    #K = 0.0030667323772751
    Y = (yy-600000.)/1000000.
    X = (xx-200000.)/1000000.
    
    a1 = (+ 4.72973056
          + 0.7925714  * X
          + 0.132812 * (X**2)
          + 0.02550 * (X**3)
          + 0.0048 * (X**4))
    a3 = (- 0.044270
          - 0.02550 * X
          - 0.0096 * (X**2))
    a5 = + 0.00096
    lmb = 2.67825 + a1*Y + a3*(Y**3) + a5*(Y**5)
    
    p0 = (  0
          + 3.23864877 * X
          - 0.0025486 * (X**2)
          - 0.013245 * (X**3)
          + 0.000048 * (X**4))
    p2 = (- 0.27135379
          - 0.0450442 * X
          - 0.007553 * (X**2)
          - 0.00146 * (X**3))
    p4 = 0.002442 + 0.00132 * X
    phi = 16.902866 + p0 + p2*(Y**2) + p4*(Y**4)
    
    phi = phi * 10000./3600. * pi/180
    lmb = lmb * 10000./3600. * pi/180
    
    return (lmb,phi)

def distInMeter(E1, N1, E2, N2):
    E1 = radians(E1)
    N1 = radians(N1)
    E2 = radians(E2)
    N2 = radians(N2)
    acosarg = sin(N1)*sin(N2) + cos(N1)*cos(N2)*cos(E1-E2)
    acosarg = min(1,acosarg)
    acosarg = max(-1, acosarg)
    return acos(acosarg) * R_earth


if __name__ == "__main__":
    from math import *
    print ("Rigi:", map(degrees, SwissReverse(212273,679520)))
